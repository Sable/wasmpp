#ifndef WASM_WASM_INSTRUCTIONS_H_
#define WASM_WASM_INSTRUCTIONS_H_

#include <src/ir.h>
#include <src/wasmpp/wasm-manager.h>

namespace wasmpp {

// Helpers
wabt::ExprList ExprToExprList(std::unique_ptr<wabt::Expr> expr);
void Merge(wabt::ExprList* e1, wabt::ExprList* e2);

// Create block expressions
wabt::ExprList CreateLoop(ModuleManager* mm, std::function<void(BlockBody, wabt::Var)> content);
wabt::ExprList CreateBlock(ModuleManager* mm, std::function<void(wabt::ExprList*, wabt::Var)> content);

// Create arithmetic expression
wabt::ExprList CreateUnary(wabt::Opcode opcode, wabt::ExprList* op);
wabt::ExprList CreateBinary(wabt::Opcode opcode, wabt::ExprList* op1, wabt::ExprList* op2);

// Create constants
wabt::ExprList CreateI32Const(uint32_t val);
wabt::ExprList CreateI64Const(uint64_t val);
wabt::ExprList CreateF32Const(float val);
wabt::ExprList CreateF64Const(double val);

  // Create loads
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
  wabt::ExprList Create##opcode(wabt::ExprList* index, wabt::Address align = 0, uint32_t offset = 0);
  LOAD_INSTRUCTIONS_LIST(DECLARE_LOAD)
#undef DECLARE_LOAD

  // Create stores
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
  wabt::ExprList Create##opcode(wabt::ExprList* index, wabt::ExprList* val, wabt::Address align = 0,  \
      uint32_t offset = 0);
  STORE_INSTRUCTIONS_LIST(DECLARE_STORE)
#undef DECLARE_STORE

// Create branch
wabt::ExprList CreateBr(wabt::Var label);
wabt::ExprList CreateBrIf(wabt::Var label, wabt::ExprList* cond);

// Create locals
wabt::ExprList CreateLocalGet(wabt::Var var);
wabt::ExprList CreateLocalSet(wabt::Var var, wabt::ExprList* val);
wabt::ExprList CreateLocalTree(wabt::Var var, wabt::ExprList* val);

// Create calls
wabt::ExprList CreateCall(wabt::Var var, std::vector<wabt::ExprList*> args);

}

#endif
