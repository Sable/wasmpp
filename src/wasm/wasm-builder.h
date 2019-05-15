#ifndef WASM_WASM_BUILDER_H_
#define WASM_WASM_BUILDER_H_

#include <src/ir.h>

namespace wasm {

class ModuleBuilder {
private:
  wabt::Module module_;
  int uid_ = 0;

  // Generate a unique id
  std::string GenerateUid();

public:

  // Append e2 to the end of e1
  static void Merge(wabt::ExprList* e1, wabt::ExprList* e2);

  // To ExprList
  static wabt::ExprList ExprToExprList(std::unique_ptr<wabt::Expr> expr);

  const wabt::Module& GetModule() const;

  // Create a new function in a module
  void CreateFunction(std::string name, wabt::FuncSignature sig, wabt::TypeVector locals,
                      std::function<void(wabt::ExprList*, std::vector<wabt::Var>, std::vector<wabt::Var>)> content);

  // Create block expressions
  wabt::ExprList CreateLoop(std::function<void(wabt::ExprList*, wabt::Var)> content);
  wabt::ExprList CreateBlock(std::function<void(wabt::ExprList*, wabt::Var)> content);

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

  // Generate increment
  wabt::ExprList GenerateIncrement(wabt::Var var, wabt::ExprList* inc, bool tee);
  
  // Generate
  //   get_local {comp_lhs}
  //   {lhs_inc_amount}
  //   i32.add
  //   tee_local {comp_lhs}
  //   {comp_rhs}
  //   {comp_op}
  //   br_if
  wabt::ExprList GenerateBranchIfCompInc(wabt::Var label, wabt::Opcode comp_op, wabt::Var comp_lhs,
                                         wabt::ExprList* lhs_inc_amount, wabt::ExprList* comp_rhs);
};

} // namespace wasm

#endif
