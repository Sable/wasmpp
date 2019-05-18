#ifndef WASM_WASM_INSTRUCTIONS_H_
#define WASM_WASM_INSTRUCTIONS_H_

#include <src/ir.h>

namespace wasmpp {

class ContentManager;
typedef ContentManager BlockBody;
typedef std::shared_ptr<wabt::ExprList> exprs_sptr;

// Helpers
exprs_sptr CreateExprList();
exprs_sptr ExprToExprList(std::unique_ptr<wabt::Expr> expr);
void Merge(exprs_sptr e1, exprs_sptr e2);

// Make block expressions
exprs_sptr MakeLoop(ContentManager* ctn, std::function<void(BlockBody, wabt::Var)> content);
exprs_sptr MakeBlock(ContentManager* ctn, std::function<void(BlockBody, wabt::Var)> content);

// Make arithmetic expression
exprs_sptr MakeUnary(wabt::Opcode opcode, exprs_sptr op);
exprs_sptr MakeBinary(wabt::Opcode opcode, exprs_sptr op1,
                                           exprs_sptr op2);

// Make constants

exprs_sptr MakeI32Const(uint32_t val);
exprs_sptr MakeI64Const(uint64_t val);
exprs_sptr MakeF32Const(float val);
exprs_sptr MakeF64Const(double val);

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
  exprs_sptr Make##opcode(exprs_sptr index,  \
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
  exprs_sptr Make##opcode(exprs_sptr index, \
  exprs_sptr val, wabt::Address align = wabt::WABT_USE_NATURAL_ALIGNMENT,  \
  uint32_t offset = 0);
  STORE_INSTRUCTIONS_LIST(DECLARE_STORE)
#undef DECLARE_STORE

// Make branch
exprs_sptr MakeBr(wabt::Var label);
exprs_sptr MakeBrIf(wabt::Var label, exprs_sptr cond);

// Make locals
exprs_sptr MakeLocalGet(wabt::Var var);
exprs_sptr MakeLocalSet(wabt::Var var, exprs_sptr val);
exprs_sptr MakeLocalTree(wabt::Var var, exprs_sptr val);

// Make calls
exprs_sptr MakeCall(wabt::Var var, std::vector<exprs_sptr> args);

}

#endif
