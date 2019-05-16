#ifndef WASM_WASM_INSTRUCTIONS_GEN_H_
#define WASM_WASM_INSTRUCTIONS_GEN_H_

#include <src/ir.h>
#include <src/wasmpp/wasm-instructions.h>

namespace wasmpp {

// Generate
//   get_local {var}
//   {inc}
//   [i32,i64,f32,f64].add
//   set_local {var} | tee_local {var}
wabt::ExprList GenerateIncrement(wabt::Type type, wabt::Var var, wabt::ExprList* inc, bool tee);

// Generate
//   get_local {comp_lhs}
//   {lhs_inc_amount}
//   [i32,i64,f32,f64].add
//   tee_local {comp_lhs}
//   {comp_rhs}
//   {comp_op}
//   br_if
wabt::ExprList GenerateBranchIfCompInc(wabt::Var label, wabt::Type type, wabt::Opcode comp_op, wabt::Var comp_lhs,
                                         wabt::ExprList* lhs_inc_amount, wabt::ExprList* comp_rhs);

} // namespace wasmpp

#endif