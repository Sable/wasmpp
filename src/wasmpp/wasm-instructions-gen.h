#ifndef WASM_WASM_INSTRUCTIONS_GEN_H_
#define WASM_WASM_INSTRUCTIONS_GEN_H_

#include <src/ir.h>
#include <src/wasmpp/wasm-instructions.h>

namespace wasmpp {

// Generate
//   i32.const {start}
//   set_local {var}
//   loop {label}
//   {content}
//   get_local {var}
//   i32.const {inc}
//   i32.add
//   tee_local {var}
//   i32.const {end}
//   i32.ne
//   br_if {label}
//   end
wabt::ExprList* GenerateRangeLoop(LabelManager* label_manager, wabt::Var var, uint32_t start, uint32_t end, uint32_t inc,
                                  wabt::FuncSignature sig, std::function<void(BlockBody*)> content);

// Generate
//   loop {label}
//   {content}
//   get_local {var}
//   i32.const {inc}
//   i32.add
//   tee_local {var}
//   get_local {end}
//   i32.ne
//   br_if {label}
//   end
wabt::ExprList* GenerateDoWhileLoop(LabelManager* label_manager, wabt::Var begin, wabt::Var end, uint32_t inc,
                                  wabt::FuncSignature sig, std::function<void(BlockBody*)> content);

// Generate
//   get_local {var}
//   {operand}
//   {op}
//   set_local {var}
wabt::ExprList* GenerateCompoundAssignment(wabt::Var var, wabt::Opcode op, wabt::ExprList* operand);

} // namespace wasmpp

#endif