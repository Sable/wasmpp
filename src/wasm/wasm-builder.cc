#include <src/wasm/wasm-builder.h>
#include <sstream>

namespace wasm {

const wabt::Module& ModuleBuilder::GetModule() const {
  return module_;
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

void ModuleBuilder::CreateFunction(std::string name, wabt::ExprList* exprList) {

  // Create a function field
  module_.AppendField(std::move(wabt::MakeUnique<wabt::FuncModuleField>()));
  Merge(&module_.funcs.back()->exprs, exprList);

  // Create an export field
  wabt::ModuleFieldList export_fields;
  auto export_field = wabt::MakeUnique<wabt::ExportModuleField>();
  export_field->export_.kind = wabt::ExternalKind::Func;
  export_field->export_.name = std::move(name);
  export_field->export_.var = wabt::Var(module_.funcs.size() - 1);
  export_fields.push_back(std::move(export_field));
  module_.AppendFields(&export_fields);
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
  wabt::ExprList e;
  e.push_back(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I32(val)));
  return e;
}

wabt::ExprList ModuleBuilder::CreateI64Const(uint64_t val) {
  wabt::ExprList e;
  e.push_back(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I64(val)));
  return e;
}

wabt::ExprList ModuleBuilder::CreateF32Const(float val) {
  wabt::ExprList e;
  uint32_t value;
  memcpy(&val, &value, sizeof(val));
  e.push_back(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F32(value)));
  return e;
}

wabt::ExprList ModuleBuilder::CreateF64Const(double val) {
  wabt::ExprList e;
  uint64_t value;
  memcpy(&val, &value, sizeof(val));
  e.push_back(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F64(value)));
  return e;
}

wabt::ExprList ModuleBuilder::CreateBr(wabt::Var label) {
  wabt::ExprList e;
  e.push_back(wabt::MakeUnique<wabt::BrExpr>(label));
  return e;
}

wabt::ExprList ModuleBuilder::CreateBrIf(wabt::Var label, wabt::ExprList* cond) {
  wabt::ExprList e;
  Merge(&e, cond);
  e.push_back(wabt::MakeUnique<wabt::BrIfExpr>(label));
  return e;
}

wabt::ExprList ModuleBuilder::CreateLocalGet(wabt::Var var) {
  wabt::ExprList e;
  e.push_back(wabt::MakeUnique<wabt::LocalGetExpr>(var));
  return e;
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

wabt::ExprList ModuleBuilder::GenerateI32Increment(wabt::Var var, uint32_t val, bool tee) {
  auto lhs = CreateLocalGet(var);
  auto rhs = CreateI32Const(val);
  auto add = CreateBinary(wabt::Opcode::I32Add, &lhs, &rhs);
  return tee ? CreateLocalTree(var, &add) : CreateLocalSet(var, &add);
}

wabt::ExprList ModuleBuilder::GenerateBranchIfCompInc(wabt::Var var, wabt::Opcode opcode, wabt::Var lhs, 
    uint32_t inc, wabt::ExprList* rhs) {
  auto tee = GenerateI32Increment(lhs, inc, true);
  auto comp = CreateBinary(opcode, &tee, rhs);
  return CreateBrIf(var, &comp);
}

} // namespace wasm

