#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions.h>
#include <src/wat-writer.h>
#include <src/binary-writer.h>
#include <src/stream.h>
#include <src/cast.h>
#include <sstream>
#include <stack>
#include <algorithm>

namespace wasmpp {

void ContentManager::Insert(exprs_sptr e) {
  while(e->size() > 0) {
    expr_list->push_back(e->extract_front());
  }
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

std::string ModuleManager::GenerateUid() {
  std::stringstream ss;
  ss << "$" << uid_++;
  return ss.str();
}

wabt::Var ModuleManager::MakeFunction(const char* name, wabt::FuncSignature sig, wabt::TypeVector locals,
                                   std::function<void(FuncBody, std::vector<wabt::Var>,
                                                      std::vector<wabt::Var>)> content) {
  // Create a function field
  wabt::Var func_name = wabt::Var(GenerateUid());
  module_.AppendField(std::move(wabt::MakeUnique<wabt::FuncModuleField>(wabt::Location(), func_name.name())));
  auto func = module_.funcs.back();
  func->decl.sig = sig;

  // Create params
  std::vector<wabt::Var> param_vars;
  for(wabt::Index i=0; i < func->GetNumParams(); i++) {
    std::string uid = GenerateUid();
    func->bindings.emplace(uid, wabt::Binding(wabt::Location(), i));
    param_vars.emplace_back(wabt::Var(uid));
  }

  // Create locals
  std::vector<wabt::Var> local_vars;
  std::vector<wabt::Type> local_types;
  for(wabt::Index i=0; i < locals.size(); i++) {
    std::string uid = GenerateUid();
    func->bindings.emplace(uid, wabt::Binding(wabt::Location(), func->GetNumParams() + i));
    local_vars.emplace_back(wabt::Var(uid));
    local_types.emplace_back(locals[i]);
  }
  func->local_types.Set(local_types);

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
  FuncBody func_body;
  func_body.expr_list = &func->exprs;
  content(func_body, param_vars, local_vars);
  return func_name;
}

wabt::Var ModuleManager::MakeFuncImport(std::string module, std::string function, wabt::FuncSignature sig) {
  wabt::Var import_name(GenerateUid());
  auto import = wabt::MakeUnique<wabt::FuncImport>(import_name.name());
  import->func.decl.sig = std::move(sig);
  auto field = wabt::MakeUnique<wabt::ImportModuleField>(std::move(import));
  field->import->module_name = std::move(module);
  field->import->field_name = std::move(function);
  module_.AppendField(std::move(field));
  return import_name;
}

wabt::Var ModuleManager::MakeMemory(uint64_t init_page, uint64_t max, bool shared) {
  wabt::Var memory_name(GenerateUid());
  auto field = wabt::MakeUnique<wabt::MemoryModuleField>(wabt::Location(), memory_name.name());
  field->memory.page_limits.initial = init_page;
  field->memory.page_limits.is_shared = shared;
  field->memory.page_limits.max = max;
  field->memory.page_limits.has_max = shared || max != 0;
  module_.AppendField(std::move(field));
  return memory_name;
}

} // namespace wasm

