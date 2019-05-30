#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/common.h>
#include <src/wat-writer.h>
#include <src/binary-writer.h>
#include <src/stream.h>
#include <src/cast.h>
#include <src/validator.h>
#include <src/resolve-names.h>
#include <sstream>
#include <stack>
#include <algorithm>

namespace wasmpp {

void ContentManager::Insert(wabt::ExprList* e) {
  Merge(expr_list_, e);
}

ContentManager::ContentManager(LabelManager* label_manager, wabt::ExprList *expr_list) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  ERROR_UNLESS(expr_list != nullptr, "expr_list cannot be null");
  label_manager_ = label_manager;
  expr_list_ = expr_list;
}

Memory::Memory(uint64_t begin, uint64_t end) {
  begin_ = begin;
  end_ = end;

  ERROR_UNLESS(begin < end, "begin must be strictly less than end");
}

MemoryManager::~MemoryManager() {
  for(size_t i=0; i < memories_.size(); i++) {
    delete memories_[i];
  }
}

uint64_t MemoryManager::Pages() {
  if(memories_.empty()) return 0;
  uint64_t val = memories_.back()->End() / WABT_PAGE_SIZE;
  return val * WABT_PAGE_SIZE == memories_.back()->End() ? val : val + 1;
}

Memory* MemoryManager::Allocate(uint64_t k) {
  ERROR_UNLESS(k > 0, "k must be positive");
  uint64_t start = 0;
  size_t i;
  for(i=0; i < memories_.size(); i++) {
    if(memories_[i]->Begin() - start >= k) {
      break;
    }
    start = memories_[i]->End();
  }
  auto memory = new Memory{start, start + k};
  memories_.insert(memories_.begin() + i, memory);
  return memory;
}

bool MemoryManager::Free(const wasmpp::Memory *m) {
  auto find = std::find(memories_.begin(), memories_.end(), m);
  if (find != memories_.end()) {
    memories_.erase(find);
    delete m;
    return true;
  }
  return false;
}

const wabt::Module& ModuleManager::GetModule() const {
  return module_;
}

bool ModuleManager::Validate() {
  wabt::Errors errors;
  wabt::ValidateOptions options;
  if(wabt::Succeeded(wabt::ResolveNamesModule(&module_, &errors))) {
    if(wabt::Succeeded(wabt::ValidateModule(&module_, &errors, options))) {
      return true;
    }
  }
  for(auto error : errors) {
    fprintf(stderr, "%s\n", error.message.c_str());
  }
  return false;
}

std::string ModuleManager::ToWat(bool folded, bool inline_import_export) const {
  wabt::WriteWatOptions wat_options;
  wat_options.fold_exprs = folded;
  wat_options.inline_import = inline_import_export;
  wat_options.inline_export = inline_import_export;
  wabt::MemoryStream stream;
  wabt::WriteWat(&stream, &module_, wat_options);
  return std::string(stream.output_buffer().data.begin(), stream.output_buffer().data.end());
}

wabt::OutputBuffer ModuleManager::ToWasm() const {
  wabt::WriteBinaryOptions binaryOptions;
  wabt::MemoryStream stream;
  WriteBinaryModule(&stream, &module_, binaryOptions);
  return stream.output_buffer();
}

std::string LabelManager::Next() {
  std::stringstream ss;
  ss << "$" << uid_++;
  return ss.str();
}

void ModuleManager::CheckImportOrdering() {
  if (module_.funcs.size() != module_.num_func_imports ||
      module_.tables.size() != module_.num_table_imports ||
      module_.memories.size() != module_.num_memory_imports ||
      module_.globals.size() != module_.num_global_imports ||
      module_.events.size() != module_.num_event_imports) {
    ERROR_EXIT("imports must occur before all non-import definitions\n")
  }
}

void ModuleManager::ResolveImplicitlyDefinedFunctionType(const wabt::FuncDeclaration& decl) {
  // Resolve implicitly defined function types, e.g.: (func (param i32) ...)
  if (!decl.has_func_type) {
    wabt::Index func_type_index = module_.GetFuncTypeIndex(decl.sig);
    if (func_type_index == wabt::kInvalidIndex) {
      auto func_type_field = wabt::MakeUnique<wabt::FuncTypeModuleField>();
      func_type_field->func_type.sig = decl.sig;
      module_.AppendField(std::move(func_type_field));
    }
  }
}

wabt::Var ModuleManager::MakeFunction(const char* name, wabt::FuncSignature sig, wabt::TypeVector locals,
                                   std::function<void(FuncBody, std::vector<wabt::Var>,
                                                      std::vector<wabt::Var>)> content) {
  // Create a function field
  wabt::Var func_name = wabt::Var(label_manager_.Next());
  auto field = wabt::MakeUnique<wabt::FuncModuleField>(wabt::Location(), func_name.name());
  field->func.decl.sig = sig;

  // Create params
  std::vector<wabt::Var> param_vars;
  for(wabt::Index i=0; i < field->func.GetNumParams(); i++) {
    std::string uid = label_manager_.Next();
    field->func.bindings.emplace(uid, wabt::Binding(wabt::Location(), i));
    param_vars.emplace_back(wabt::Var(uid));
  }

  // Create locals
  std::vector<wabt::Var> local_vars;
  std::vector<wabt::Type> local_types;
  for(wabt::Index i=0; i < locals.size(); i++) {
    std::string uid = label_manager_.Next();
    field->func.bindings.emplace(uid, wabt::Binding(wabt::Location(), field->func.GetNumParams() + i));
    local_vars.emplace_back(wabt::Var(uid));
    local_types.emplace_back(locals[i]);
  }
  field->func.local_types.Set(local_types);
  FuncBody func_body(&label_manager_, &field->func.exprs);
  ResolveImplicitlyDefinedFunctionType(field->func.decl);
  module_.AppendField(std::move(field));

  // Create an export field
  if(name) {
    wabt::ModuleFieldList export_fields;
    auto export_field = wabt::MakeUnique<wabt::ExportModuleField>();
    export_field->export_.kind = wabt::ExternalKind::Func;
    export_field->export_.name = name;
    export_field->export_.var = wabt::Var(module_.funcs.size() - 1);
    export_fields.push_back(std::move(export_field));
    module_.AppendFields(&export_fields);
  }

  // Populate content
  content(func_body, param_vars, local_vars);
  return func_name;
}

wabt::Var ModuleManager::MakeFuncImport(std::string module, std::string function, wabt::FuncSignature sig) {
  CheckImportOrdering();
  wabt::Var import_name(label_manager_.Next());
  auto import = wabt::MakeUnique<wabt::FuncImport>(import_name.name());
  import->func.decl.sig = std::move(sig);
  ResolveImplicitlyDefinedFunctionType(import->func.decl);
  auto field = wabt::MakeUnique<wabt::ImportModuleField>(std::move(import));
  field->import->module_name = std::move(module);
  field->import->field_name = std::move(function);
  module_.AppendField(std::move(field));
  return import_name;
}

wabt::Var ModuleManager::MakeMemory(uint64_t init_page, uint64_t max, bool shared) {
  wabt::Var memory_name(label_manager_.Next());
  auto field = wabt::MakeUnique<wabt::MemoryModuleField>(wabt::Location(), memory_name.name());
  field->memory.page_limits.initial = init_page;
  field->memory.page_limits.is_shared = shared;
  field->memory.page_limits.max = max;
  field->memory.page_limits.has_max = shared || max != 0;
  module_.AppendField(std::move(field));
  return memory_name;
}

void ModuleManager::MakeData(wabt::Var var, uint32_t index, std::vector<wasmpp::DataEntry> entries) {
  assert(var.type() == wabt::VarType::Name);
  auto field = wabt::MakeUnique<wabt::DataSegmentModuleField>(wabt::Location(), var.name());
  field->data_segment.memory_var = var;
  field->data_segment.offset.splice(field->data_segment.offset.end(), *MakeI32Const(index));

  // Insert data in little endian
  std::vector<uint8_t> data;
  for(auto entry : entries) {
    uint64_t value_bits;
    uint64_t mask = 0x00000000000000ff;
    if(entry.kind == DataEntry::Kind::I32) {
      memcpy(&value_bits, &entry.val.i32, entry.Size());
    } else if(entry.kind == DataEntry::Kind::I64) {
      memcpy(&value_bits, &entry.val.i64, entry.Size());
    } else if(entry.kind == DataEntry::Kind::F32) {
      memcpy(&value_bits, &entry.val.f32, entry.Size());
    } else if(entry.kind == DataEntry::Kind::F64) {
      memcpy(&value_bits, &entry.val.f64, entry.Size());
    } else {
      assert(entry.kind == DataEntry::Kind::Byte);
      memcpy(&value_bits, &entry.val.f64, entry.Size());
    }
    for(int i=0; i < entry.Size(); i++) {
      data.push_back((uint8_t) (mask & value_bits));
      value_bits >>= 8;
    }
  }
  field->data_segment.data = data;
  module_.AppendField(std::move(field));
}

void ModuleManager::MakeExport(std::string name, wabt::Var var, wabt::ExternalKind kind) {
  auto field = wabt::MakeUnique<wabt::ExportModuleField>();
  field->export_.name = name;
  field->export_.kind = kind;
  field->export_.var = var;
  module_.AppendField(std::move(field));
}

void ModuleManager::MakeMemoryExport(std::string name, wabt::Var var) {
  MakeExport(name, var, wabt::ExternalKind::Memory);
}

} // namespace wasm

