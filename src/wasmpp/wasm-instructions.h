#ifndef WASM_WASM_INSTRUCTIONS_H_
#define WASM_WASM_INSTRUCTIONS_H_

#include <src/ir.h>
#include <src/wasmpp/common.h>

namespace wasmpp {

class LabelManager;
class ContentManager;
typedef ContentManager BlockBody;

// Helpers
wabt::ExprList* ExprToExprList(std::unique_ptr<wabt::Expr> expr);
void Merge(wabt::ExprList* e1, wabt::ExprList* e2);

// Make block expressions
wabt::ExprList* MakeLoop(LabelManager* label_manager, wabt::FuncSignature sig,
                         std::function<void(BlockBody, wabt::Var)> content);
wabt::ExprList* MakeBlock(LabelManager* label_manager, wabt::FuncSignature sig,
                          std::function<void(BlockBody, wabt::Var)> content);
wabt::ExprList* MakeIf(LabelManager* label_manager, wabt::ExprList* cond, wabt::FuncSignature sig,
                       std::function<void(BlockBody, wabt::Var)> true_content,
                       std::function<void(BlockBody)> false_content = {});

// Make arithmetic expression
wabt::ExprList* MakeUnary(wabt::Opcode opcode, wabt::ExprList* op);
wabt::ExprList* MakeBinary(wabt::Opcode opcode, wabt::ExprList* op1, wabt::ExprList* op2);

// Make constants
wabt::ExprList* MakeI32Const(uint32_t val);
wabt::ExprList* MakeI64Const(uint64_t val);
wabt::ExprList* MakeF32Const(float val);
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

// Make branch
wabt::ExprList* MakeBr(wabt::Var label);
wabt::ExprList* MakeBrIf(wabt::Var label, wabt::ExprList* cond);

// Make locals
wabt::ExprList* MakeLocalGet(wabt::Var var);
wabt::ExprList* MakeLocalSet(wabt::Var var, wabt::ExprList* val);
wabt::ExprList* MakeLocalTree(wabt::Var var, wabt::ExprList* val);

// Make calls
wabt::ExprList* MakeCall(wabt::Var var, std::vector<wabt::ExprList*> args);

// Misc
wabt::ExprList* MakeDrop();
wabt::ExprList* MakeNop();

}

#endif
