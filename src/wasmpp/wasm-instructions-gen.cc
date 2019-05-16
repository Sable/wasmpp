#include <src/wasmpp/wasm-instructions-gen.h>

namespace wasmpp {

wabt::ExprList GenerateIncrement(wabt::Type type, wabt::Var var, wabt::ExprList *inc, bool tee) {
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

wabt::ExprList GenerateBranchIfCompInc(wabt::Var label, wabt::Type type, wabt::Opcode comp_op,
                                       wabt::Var comp_lhs, wabt::ExprList *lhs_inc_amount,
                                       wabt::ExprList *comp_rhs) {
  auto tee = GenerateIncrement(type, std::move(comp_lhs), lhs_inc_amount, true);
  auto comp = CreateBinary(comp_op, &tee, comp_rhs);
  return CreateBrIf(std::move(label), &comp);
}

} // namespace wasmpp