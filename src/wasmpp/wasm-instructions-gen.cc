#include <src/wasmpp/wasm-instructions-gen.h>
#include <utility>

namespace wasmpp {

exprs_sptr GenerateIncrement(wabt::Type type, wabt::Var var, exprs_sptr inc, bool tee) {
  auto lhs = MakeLocalGet(var);
  wabt::Opcode op;
  switch(type) {
    case wabt::Type::I32:
      op = wabt::Opcode::I32Add;
      break;
    case wabt::Type::I64:
      op = wabt::Opcode::I64Add;
      break;
    case wabt::Type::F32:
      op = wabt::Opcode::F32Add;
      break;
    case wabt::Type::F64:
      op = wabt::Opcode::F64Add;
      break;
    default:
      assert(!"Type not supported");
  }
  exprs_sptr add = MakeBinary(op, lhs, inc);
  return tee ? MakeLocalTree(var, add) : MakeLocalSet(var, add);
}

exprs_sptr GenerateBranchIfCompInc(wabt::Var label, wabt::Type type, wabt::Opcode comp_op,
                                       wabt::Var comp_lhs, exprs_sptr lhs_inc_amount, exprs_sptr comp_rhs) {
  auto tee = GenerateIncrement(type, std::move(comp_lhs), lhs_inc_amount, true);
  auto comp = MakeBinary(comp_op, tee, comp_rhs);
  return MakeBrIf(std::move(label), comp);
}

} // namespace wasmpp