#include <src/wasmpp/wasm-builder.h>
#include <src/wat-writer.h>
#include <src/binary-writer.h>
#include <src/stream.h>
#include <sstream>

namespace wasmpp {

const wabt::Module& ModuleBuilder::GetModule() const {
  return module_;
}

std::string ModuleBuilder::ToWat(bool folded, bool inline_import_export) const {
  wabt::WriteWatOptions wat_options;
  wat_options.fold_exprs = folded;
  wat_options.inline_import = inline_import_export;
  wat_options.inline_export = inline_import_export;
  wabt::MemoryStream stream;
  wabt::WriteWat(&stream, &module_, wat_options);
  return std::string(stream.output_buffer().data.begin(), stream.output_buffer().data.end());
}

wabt::OutputBuffer ModuleBuilder::ToWasm() const {
  wabt::WriteBinaryOptions binaryOptions;
  wabt::MemoryStream stream;
  WriteBinaryModule(&stream, &module_, binaryOptions);
  return stream.output_buffer();
}

std::string ModuleBuilder::GenerateUid() {
  std::stringstream ss;
  ss << "$" << uid_++;
  return ss.str();
}

void ModuleBuilder::Merge(wabt::ExprList* e1, wabt::ExprList* e2) {
  while(e2->size() > 0) {
    e1->push_back(e2->extract_front());
  }
}

wabt::ExprList ModuleBuilder::ExprToExprList(std::unique_ptr<wabt::Expr> expr) {
  wabt::ExprList e;
  e.push_back(std::move(expr));
  return e;
}

wabt::Var ModuleBuilder::CreateFunction(const char* name, wabt::FuncSignature sig, wabt::TypeVector locals,
                                   std::function<void(wabt::ExprList *, std::vector<wabt::Var>,
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
    export_field->export_.name = std::move(name);
    export_field->export_.var = wabt::Var(module_.funcs.size() - 1);
    export_fields.push_back(std::move(export_field));
    module_.AppendFields(&export_fields);
  }

  // Populate content
  content(&func->exprs, param_vars, local_vars);
  return func_name;
}

wabt::Var ModuleBuilder::CreateFuncImport(std::string module, std::string function, wabt::FuncSignature sig) {
  wabt::Var import_name(GenerateUid());
  auto import = wabt::MakeUnique<wabt::FuncImport>(import_name.name());
  import->func.decl.sig = sig;
  auto field = wabt::MakeUnique<wabt::ImportModuleField>(std::move(import));
  field->import->module_name = module;
  field->import->field_name = function;
  module_.AppendField(std::move(field));
  return import_name;
}

wabt::Var ModuleBuilder::CreateMemory(uint64_t init_page, uint64_t max, bool shared) {
  wabt::Var memory_name(GenerateUid());
  auto field = wabt::MakeUnique<wabt::MemoryModuleField>(wabt::Location(), memory_name.name());
  field->memory.page_limits.initial = init_page;
  field->memory.page_limits.is_shared = shared;
  field->memory.page_limits.max = max;
  if(shared || max != 0) {
    field->memory.page_limits.has_max = true;
  } else {
    field->memory.page_limits.has_max = false;
  }
  module_.AppendField(std::move(field));
  return memory_name;
}

wabt::ExprList ModuleBuilder::CreateLoop(std::function<void(wabt::ExprList*, wabt::Var)> content) {
  wabt::ExprList e;
  auto loop = wabt::MakeUnique<wabt::LoopExpr>();
  loop->block.label = GenerateUid();
  content(&loop->block.exprs, wabt::Var(loop->block.label));
  e.push_back(std::move(loop));
  return e;
}

wabt::ExprList ModuleBuilder::CreateBlock(std::function<void(wabt::ExprList*, wabt::Var)> content) {
  wabt::ExprList e;
  auto block = wabt::MakeUnique<wabt::LoopExpr>();
  block->block.label = GenerateUid();
  content(&block->block.exprs, wabt::Var(block->block.label));
  e.push_back(std::move(block));
  return e;
}

wabt::ExprList ModuleBuilder::CreateUnary(wabt::Opcode opcode, wabt::ExprList* op) {
  wabt::ExprList e;
  Merge(&e, op);
  e.push_back(wabt::MakeUnique<wabt::UnaryExpr>(opcode));
  return e;
}

wabt::ExprList ModuleBuilder::CreateBinary(wabt::Opcode opcode, wabt::ExprList* op1,
                                                            wabt::ExprList* op2) {
  wabt::ExprList e;
  Merge(&e, op1);
  Merge(&e, op2);
  e.push_back(wabt::MakeUnique<wabt::BinaryExpr>(opcode));
  return e;
}

wabt::ExprList ModuleBuilder::CreateI32Const(uint32_t val) {
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I32(val)));
}

wabt::ExprList ModuleBuilder::CreateI64Const(uint64_t val) {
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I64(val)));
}

wabt::ExprList ModuleBuilder::CreateF32Const(float val) {
  uint32_t value;
  memcpy(&val, &value, sizeof(val));
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F32(value)));
}

wabt::ExprList ModuleBuilder::CreateF64Const(double val) {
  uint64_t value;
  memcpy(&val, &value, sizeof(val));
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F64(value)));
}

wabt::ExprList ModuleBuilder::CreateBr(wabt::Var label) {
  return ExprToExprList(wabt::MakeUnique<wabt::BrExpr>(label));
}

wabt::ExprList ModuleBuilder::CreateBrIf(wabt::Var label, wabt::ExprList* cond) {
  wabt::ExprList e;
  Merge(&e, cond);
  e.push_back(wabt::MakeUnique<wabt::BrIfExpr>(label));
  return e;
}

wabt::ExprList ModuleBuilder::CreateLocalGet(wabt::Var var) {
  return ExprToExprList(wabt::MakeUnique<wabt::LocalGetExpr>(var));
}

wabt::ExprList ModuleBuilder::CreateLocalSet(wabt::Var var, wabt::ExprList* val) {
  wabt::ExprList e;
  Merge(&e, val);
  e.push_back(wabt::MakeUnique<wabt::LocalSetExpr>(var));
  return e;
}

wabt::ExprList ModuleBuilder::CreateLocalTree(wabt::Var var, wabt::ExprList* val) {
  wabt::ExprList e;
  Merge(&e, val);
  e.push_back(wabt::MakeUnique<wabt::LocalTeeExpr>(var));
  return e;
}

wabt::ExprList ModuleBuilder::CreateCall(wabt::Var var, std::vector<wabt::ExprList*> args) {
  wabt::ExprList e;
  for(auto arg : args) {
    Merge(&e, arg);
  }
  e.push_back(wabt::MakeUnique<wabt::CallExpr>(var));
  return e;
}

#define DEFINE_LOAD(opcode) \
wabt::ExprList ModuleBuilder::Create##opcode(wabt::ExprList *index, wabt::Address align, uint32_t offset) { \
  wabt::ExprList e;                                                                                         \
  Merge(&e, index);                                                                                         \
  e.push_back(wabt::MakeUnique<wabt::LoadExpr>(wabt::Opcode::opcode, align, offset));                       \
  return e;                                                                                                 \
}
LOAD_INSTRUCTIONS_LIST(DEFINE_LOAD)
#undef DEFINE_LOAD

#define DEFINE_STORE(opcode) \
wabt::ExprList ModuleBuilder::Create##opcode(wabt::ExprList *index, wabt::ExprList* val,        \
    wabt::Address align, uint32_t offset) {                                                     \
  wabt::ExprList e;                                                                             \
  Merge(&e, index);                                                                             \
  Merge(&e, val);                                                                               \
  e.push_back(wabt::MakeUnique<wabt::StoreExpr>(wabt::Opcode::opcode, align, offset));          \
  return e;                                                                                     \
}
STORE_INSTRUCTIONS_LIST(DEFINE_STORE)
#undef DEFINE_STORE

wabt::ExprList ModuleBuilder::GenerateIncrement(wabt::Type type, wabt::Var var, wabt::ExprList *inc, bool tee) {
  auto lhs = CreateLocalGet(var);
  wabt::ExprList add;
  switch(type) {
    case wabt::Type::I32:
      add = CreateBinary(wabt::Opcode::I32Add, &lhs, inc);
      break;
    case wabt::Type::I64:
      add = CreateBinary(wabt::Opcode::I64Add, &lhs, inc);
      break;
    case wabt::Type::F32:
      add = CreateBinary(wabt::Opcode::F32Add, &lhs, inc);
      break;
    case wabt::Type::F64:
      add = CreateBinary(wabt::Opcode::F64Add, &lhs, inc);
      break;
    default:
      assert(!"Type not supported");
  }
  return tee ? CreateLocalTree(var, &add) : CreateLocalSet(var, &add);
}

wabt::ExprList ModuleBuilder::GenerateBranchIfCompInc(wabt::Var label, wabt::Type type, wabt::Opcode comp_op,
                                                      wabt::Var comp_lhs, wabt::ExprList *lhs_inc_amount,
                                                      wabt::ExprList *comp_rhs) {
  auto tee = GenerateIncrement(type, std::move(comp_lhs), lhs_inc_amount, true);
  auto comp = CreateBinary(comp_op, &tee, comp_rhs);
  return CreateBrIf(std::move(label), &comp);
}

} // namespace wasm

