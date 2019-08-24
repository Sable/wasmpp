/*!
 * @file wasm-instructions.h
 */

#ifndef WASM_WASM_INSTRUCTIONS_H_
#define WASM_WASM_INSTRUCTIONS_H_

#include <src/ir.h>
#include <src/wasmpp/common.h>

namespace wasmpp {

class LabelManager;
class ContentManager;
typedef ContentManager BlockBody;

/*!
 * Convert an expression into an expression list
 * with one element
 * @param expr Expression
 * @return Expression list
 */
wabt::ExprList* ExprToExprList(std::unique_ptr<wabt::Expr> expr);

/*!
 * Merge second expression list into first expression list
 * @note Expression lists will be modified
 * @param e1 Merge into list
 * @param e2 Merge from list
 */
void Merge(wabt::ExprList* e1, wabt::ExprList* e2);

/*!
 * Make a Wasm loop
 * @param label_manager Label manager
 * @param sig Loop signature
 * @param content Loop body
 * @return Expression list
 */
wabt::ExprList* MakeLoop(LabelManager* label_manager, wabt::FuncSignature sig,
                         std::function<void(BlockBody, wabt::Var)> content);

/*!
 * Make a Wasm block
 * @param label_manager Label manager
 * @param sig Block signature
 * @param content Block body
 * @return Expression list
 */
wabt::ExprList* MakeBlock(LabelManager* label_manager, wabt::FuncSignature sig,
                          std::function<void(BlockBody, wabt::Var)> content);

/*!
 * Make a Wasm if statement
 * @param label_manager Label manager
 * @param cond If condition
 * @param sig Statement signature
 * @param true_content True case block body
 * @param false_content False case block body
 * @return Expression list
 */
wabt::ExprList* MakeIf(LabelManager* label_manager, wabt::ExprList* cond, wabt::FuncSignature sig,
                       std::function<void(BlockBody, wabt::Var)> true_content,
                       std::function<void(BlockBody)> false_content = {});

/*!
 * Make a Wasm unary operation
 * @param opcode Operation
 * @param op Operand
 * @return Expression list
 */
wabt::ExprList* MakeUnary(wabt::Opcode opcode, wabt::ExprList* op);

/*!
 * Make a Wasm binary operation
 * @param opcode Operation
 * @param op1 Left operand
 * @param op2 Right operand
 * @return Expression list
 */
wabt::ExprList* MakeBinary(wabt::Opcode opcode, wabt::ExprList* op1, wabt::ExprList* op2);

/*!
 * Make a Wasm i32 constant
 * @param val Value
 * @return Expression list
 */
wabt::ExprList* MakeI32Const(uint32_t val);

/*!
 * Make a Wasm i64 constant
 * @param val Value
 * @return Expression list
 */
wabt::ExprList* MakeI64Const(uint64_t val);

/*!
 * Make a Wasm f32 constant
 * @param val Value
 * @return Expression list
 */
wabt::ExprList* MakeF32Const(float val);

/*!
 * Make a Wasm f64 constant
 * @param val Value
 * @return Expression list
 */
wabt::ExprList* MakeF64Const(double val);

// Make SIMD extract lane
#define EXTRACT_LANE_LIST(V) \
  V(I32X4ExtractLane) \
  V(F32X4ExtractLane) \
  V(I64X2ExtractLane) \
  V(F64X2ExtractLane)

#define DECLARE_EXTRACT(name) \
  wabt::ExprList* Make##name(wabt::ExprList* operand, uint64_t index);
EXTRACT_LANE_LIST(DECLARE_EXTRACT)
#undef DECLARE_EXTRACT

// Make SIMD replace lane
#define REPLACE_LANE_LIST(V) \
  V(I32X4ReplaceLane) \
  V(F32X4ReplaceLane) \
  V(I64X2ReplaceLane) \
  V(F64X2ReplaceLane)

#define DECLARE_REPLACE(name) \
  wabt::ExprList* Make##name(wabt::ExprList* operand, wabt::ExprList* val, uint64_t index);
  REPLACE_LANE_LIST(DECLARE_REPLACE)
#undef DECLARE_REPLACE

  // Make loads
#define LOAD_INSTRUCTIONS_LIST(V) \
  V(I32Load)    \
  V(I64Load)    \
  V(F32Load)    \
  V(F64Load)    \
  V(I32Load8S)  \
  V(I32Load8U)  \
  V(I32Load16S) \
  V(I32Load16U) \
  V(I64Load8S)  \
  V(I64Load8U)  \
  V(I64Load16S) \
  V(I64Load16U) \
  V(I64Load32S) \
  V(I64Load32U) \
  V(V128Load)

#define DECLARE_LOAD(opcode) \
  wabt::ExprList* Make##opcode(wabt::ExprList* index,  \
  wabt::Address align = wabt::WABT_USE_NATURAL_ALIGNMENT, uint32_t offset = 0);
  LOAD_INSTRUCTIONS_LIST(DECLARE_LOAD)
#undef DECLARE_LOAD

  // Make stores
#define STORE_INSTRUCTIONS_LIST(V) \
  V(I32Store)   \
  V(I64Store)   \
  V(F32Store)   \
  V(F64Store)   \
  V(I32Store8)  \
  V(I32Store16) \
  V(I64Store8)  \
  V(I64Store16) \
  V(I64Store32) \
  V(V128Store)

#define DECLARE_STORE(opcode) \
  wabt::ExprList* Make##opcode(wabt::ExprList* index, \
  wabt::ExprList* val, wabt::Address align = wabt::WABT_USE_NATURAL_ALIGNMENT,  \
  uint32_t offset = 0);
  STORE_INSTRUCTIONS_LIST(DECLARE_STORE)
#undef DECLARE_STORE

/*!
 * Make a branch instruction
 * @param label Reference variable of a loop
 * @return Expression list
 */
wabt::ExprList* MakeBr(wabt::Var label);

/*!
 * Make a branch-if instruction
 * @param label Reference variable of a loop
 * @return Expression list
 */
wabt::ExprList* MakeBrIf(wabt::Var label, wabt::ExprList* cond);

/*!
 * Make Wasm local get instruction
 * @param var Local reference variable
 * @return Expression list
 */
wabt::ExprList* MakeLocalGet(wabt::Var var);

/*!
 * Make Wasm local set instruction
 * @param var Local reference variable
 * @return Expression list
 */
wabt::ExprList* MakeLocalSet(wabt::Var var, wabt::ExprList* val);

/*!
 * Make Wasm local tee instruction
 * @param var Local reference variable
 * @return Expression list
 */
wabt::ExprList* MakeLocalTree(wabt::Var var, wabt::ExprList* val);

/*!
 * Make a function call
 * @param var Function reference variable
 * @param args List of arguments
 * @return Expression list
 */
wabt::ExprList* MakeCall(wabt::Var var, std::vector<wabt::ExprList*> args);

/*!
 * Make a Wasm drop instruction
 * @return Expression list
 */
wabt::ExprList* MakeDrop();

/*!
 * Make a Wasm nop instruction
 * @return Expression list
 */
wabt::ExprList* MakeNop();

#ifdef WABT_EXPERIMENTAL
wabt::ExprList* MakeNativeCall(wabt::Var var, std::vector<wabt::ExprList*> args);
wabt::ExprList* MakeOffset32(wabt::ExprList* base, wabt::ExprList* offset, wabt::ExprList* size);
wabt::ExprList* MakeDup(wabt::ExprList* expr);
wabt::ExprList* MakeSwap(wabt::ExprList* expr1, wabt::ExprList* expr2);
#endif

}

#endif
