#ifndef WASM_WASM_INSTRUCTIONS_H_
#define WASM_WASM_INSTRUCTIONS_H_

#include <src/ir.h>

namespace wasmpp {

class LabelManager;
class ContentManager;
typedef ContentManager BlockBody;

// Helpers
wabt::ExprList* ExprToExprList(std::unique_ptr<wabt::Expr> expr);
void Merge(wabt::ExprList* e1, wabt::ExprList* e2);

// Make block expressions
wabt::ExprList* MakeLoop(LabelManager* label_manager, std::function<void(BlockBody, wabt::Var)> content);
wabt::ExprList* MakeBlock(LabelManager* label_manager, std::function<void(BlockBody, wabt::Var)> content);
wabt::ExprList* MakeIf(LabelManager* label_manager, wabt::ExprList* cond, std::function<void(BlockBody, BlockBody,
                                                                                             wabt::Var)> content);

// Make arithmetic expression
wabt::ExprList* MakeUnary(wabt::Opcode opcode, wabt::ExprList* op);
wabt::ExprList* MakeBinary(wabt::Opcode opcode, wabt::ExprList* op1, wabt::ExprList* op2);

// Make constants

wabt::ExprList* MakeI32Const(uint32_t val);
wabt::ExprList* MakeI64Const(uint64_t val);
wabt::ExprList* MakeF32Const(float val);
wabt::ExprList* MakeF64Const(double val);

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
  V(I64Load32U)

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
  V(I64Store32)

#define DECLARE_STORE(opcode) \
  wabt::ExprList* Make##opcode(wabt::ExprList* index, \
  wabt::ExprList* val, wabt::Address align = wabt::WABT_USE_NATURAL_ALIGNMENT,  \
  uint32_t offset = 0);
  STORE_INSTRUCTIONS_LIST(DECLARE_STORE)
#undef DECLARE_STORE

// Make branch
wabt::ExprList* MakeBr(wabt::Var label);
wabt::ExprList* MakeBrIf(wabt::Var label, wabt::ExprList* cond);

// Make locals
wabt::ExprList* MakeLocalGet(wabt::Var var);
wabt::ExprList* MakeLocalSet(wabt::Var var, wabt::ExprList* val);
wabt::ExprList* MakeLocalTree(wabt::Var var, wabt::ExprList* val);

// Make calls
wabt::ExprList* MakeCall(wabt::Var var, std::vector<wabt::ExprList*> args);

}

#endif
