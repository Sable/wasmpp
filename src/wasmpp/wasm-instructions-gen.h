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
wabt::ExprList* GenerateDoWhileLoop(LabelManager* label_manager, wabt::Var begin, uint32_t end, uint32_t inc,
                                    wabt::FuncSignature sig, std::function<void(BlockBody*)> content);

// Generate
//   get_local {var}
//   {operand}
//   {op}
//   set_local {var}
wabt::ExprList* GenerateCompoundAssignment(wabt::Var var, wabt::Opcode op, wabt::ExprList* operand);

// TODO This is a temporary solution for horizontally summing v128
// Future release will include a SIMD instruction for this
// Generate
//   f32.extract_lane {var} 0
//   f32.extract_lane {var} 1
//   f32.add
//   f32.extract_lane {var} 2
//   f32.extract_lane {var} 3
//   f32.add
//   f32.add
  wabt::ExprList* GenerateExpensiveF32X4HorizontalSum(wabt::Var var);

} // namespace wasmpp

#endif