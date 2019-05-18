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

void ContentManager::Insert(exprs_sptr e) {
  while(e->size() > 0) {
    expr_list_->push_back(e->extract_front());
  }
}

ContentManager::ContentManager(wasmpp::ModuleManager *mm, wabt::ExprList *expr_list) {
  assert(mm != nullptr);
  assert(expr_list != nullptr);
  parent.module_manager_ = mm;
  parent_module_ = true;
  expr_list_ = expr_list;
}

ContentManager::ContentManager(wasmpp::ContentManager *parent_ctn, wabt::ExprList *expr_list) {
  assert(parent_ctn != nullptr);
  assert(expr_list != nullptr);
  parent.content_manager_= parent_ctn;
  parent_module_ = false;
  expr_list_ = expr_list;
}

std::string ContentManager::NextLabel() const {
  assert(HasParent());
  if(parent_module_) {
    return parent.module_manager_->NextLabel();
  }
  return parent.content_manager_->NextLabel();
}

bool ContentManager::HasParent() const {
  return parent_module_ ?
         parent.module_manager_ != nullptr :
         parent.content_manager_ != nullptr;
}

MemoryManager::~MemoryManager() {
  for(size_t i=0; i < memories_.size(); i++) {
    delete memories_[i];
  }
}

uint64_t MemoryManager::Pages() {
  if(memories_.empty()) return 0;
  uint64_t val = memories_.back()->end / WABT_PAGE_SIZE;
  return val * WABT_PAGE_SIZE == memories_.back()->end ? val : val + 1;
}

Memory* MemoryManager::Allocate(uint64_t k) {
  assert(k > 0);
  uint64_t start = 0;
  size_t i;
  for(i=0; i < memories_.size(); i++) {
    if(memories_[i]->begin - start >= k) {
      break;
    }
    start = memories_[i]->end;
  }
  auto memory = new Memory{start, start + k};
  memories_.insert(memories_.begin() + i, memory);
  return memory;
}

bool MemoryManager::Free(wasmpp::Memory *m) {
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

std::string ModuleManager::NextLabel() {
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

wabt::Var ModuleManager::MakeFunction(const char* name, wabt::FuncSignature sig, wabt::TypeVector locals,
                                   std::function<void(FuncBody, std::vector<wabt::Var>,
                                                      std::vector<wabt::Var>)> content) {
  // Create a function field
  wabt::Var func_name = wabt::Var(NextLabel());
  auto field = wabt::MakeUnique<wabt::FuncModuleField>(wabt::Location(), func_name.name());
  field->func.decl.sig = sig;

  // Create params
  std::vector<wabt::Var> param_vars;
  for(wabt::Index i=0; i < field->func.GetNumParams(); i++) {
    std::string uid = NextLabel();
    field->func.bindings.emplace(uid, wabt::Binding(wabt::Location(), i));
    param_vars.emplace_back(wabt::Var(uid));
  }

  // Create locals
  std::vector<wabt::Var> local_vars;
  std::vector<wabt::Type> local_types;
  for(wabt::Index i=0; i < locals.size(); i++) {
    std::string uid = NextLabel();
    field->func.bindings.emplace(uid, wabt::Binding(wabt::Location(), field->func.GetNumParams() + i));
    local_vars.emplace_back(wabt::Var(uid));
    local_types.emplace_back(locals[i]);
  }
  field->func.local_types.Set(local_types);
  FuncBody func_body(this, &field->func.exprs);
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
  wabt::Var import_name(NextLabel());
  auto import = wabt::MakeUnique<wabt::FuncImport>(import_name.name());
  import->func.decl.sig = std::move(sig);
  auto field = wabt::MakeUnique<wabt::ImportModuleField>(std::move(import));
  field->import->module_name = std::move(module);
  field->import->field_name = std::move(function);
  module_.AppendField(std::move(field));
  return import_name;
}

wabt::Var ModuleManager::MakeMemory(uint64_t init_page, uint64_t max, bool shared) {
  wabt::Var memory_name(NextLabel());
  auto field = wabt::MakeUnique<wabt::MemoryModuleField>(wabt::Location(), memory_name.name());
  field->memory.page_limits.initial = init_page;
  field->memory.page_limits.is_shared = shared;
  field->memory.page_limits.max = max;
  field->memory.page_limits.has_max = shared || max != 0;
  module_.AppendField(std::move(field));
  return memory_name;
}

} // namespace wasm

