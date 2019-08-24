/*!
 * @file wasm-instructions-gen.h
 */

#ifndef WASM_WASM_INSTRUCTIONS_GEN_H_
#define WASM_WASM_INSTRUCTIONS_GEN_H_

#include <src/ir.h>
#include <src/wasmpp/wasm-instructions.h>

namespace wasmpp {

/*!
 * General a Wasm loop
 * <pre>
 * i32.const {start}
 * set_local {var}
 * loop {label}
 *   {content}
 *   get_local {var}
 *   i32.const {inc}
 *   i32.add
 *   tee_local {var}
 *   i32.const {end}
 *   i32.ne
 *   br_if {label}
 * end
 * </pre>
 * @param label_manager Label manager
 * @param var Loop reference variable
 * @param start From
 * @param end To
 * @param inc Increment value
 * @param sig Loop signature
 * @param content Loop content
 * @return Expression list
 */
wabt::ExprList* GenerateRangeLoop(LabelManager* label_manager, wabt::Var var, uint32_t start, uint32_t end, uint32_t inc,
                                  wabt::FuncSignature sig, std::function<void(BlockBody*)> content);

// Generate
/*!
 * General a Wasm loop
 * <pre>
 * i32.const {start}
 * set_local {var}
 * loop {label}
 *   {content}
 *   get_local {var}
 *   get_local {inc}
 *   i32.add
 *   tee_local {var}
 *   get_local {end}
 *   i32.ne
 *   br_if {label}
 * end
 * </pre>
 * @param label_manager Label manager
 * @param var Loop reference variable
 * @param start From
 * @param end To
 * @param inc Increment value
 * @param sig Loop signature
 * @param content Loop content
 * @return Expression list
 */
wabt::ExprList* GenerateRangeLoop(LabelManager* label_manager, wabt::Var var, uint32_t start, wabt::Var end, wabt::Var inc,
                                    wabt::FuncSignature sig, std::function<void(BlockBody*)> content);

/*!
 * General a Wasm loop
 * <pre>
 *   i32.const {start}
 *   set_local {var}
 *   loop {label}
 *     {content}
 *     get_local {var}
 *     i32.const {inc}
 *     i32.add
 *     tee_local {var}
 *     get_local {end}
 *     i32.ne
 *     br_if {label}
 *   end
 * </pre>
 * @param label_manager Label manager
 * @param var Loop reference variable
 * @param start From
 * @param end To
 * @param inc Increment value
 * @param sig Loop signature
 * @param content Loop content
 * @return Expression list
 */
  wabt::ExprList* GenerateRangeLoop(LabelManager* label_manager, wabt::Var var, uint32_t start, wabt::Var end, uint32_t inc,
                                    wabt::FuncSignature sig, std::function<void(BlockBody*)> content);

/*!
 * General a Wasm loop
 * <pre>
 * loop {label}
 *   {content}
 *   get_local {var}
 *   i32.const {inc}
 *   i32.add
 *   tee_local {var}
 *   get_local {end}
 *   i32.ne
 *   br_if {label}
 * end
 * </pre>
 * @param label_manager Label manager
 * @param var Loop reference variable
 * @param begin From
 * @param end To
 * @param inc Increment value
 * @param sig Loop signature
 * @param content Loop content
 * @return Expression list
 */
wabt::ExprList* GenerateDoWhileLoop(LabelManager* label_manager, wabt::Var begin, wabt::Var end, uint32_t inc,
                                  wabt::FuncSignature sig, std::function<void(BlockBody*)> content);

/*!
 * General a Wasm loop
 * <pre>
 * loop {label}
 *   {content}
 *   get_local {var}
 *   i32.const {inc}
 *   i32.add
 *   tee_local {var}
 *   i32.const {end}
 *   i32.ne
 *   br_if {label}
 * end
 * </pre>
 * @param label_manager Label manager
 * @param var Loop reference variable
 * @param start From
 * @param end To
 * @param inc Increment value
 * @param sig Loop signature
 * @param content Loop content
 * @return Expression list
 */
wabt::ExprList* GenerateDoWhileLoop(LabelManager* label_manager, wabt::Var begin, uint32_t end, uint32_t inc,
                                    wabt::FuncSignature sig, std::function<void(BlockBody*)> content);

/*!
 * Generate a Wasm compunt assignment
 * @param var Reference variable
 * @param op Operation
 * @param operand Operand
 * @return Expression list
 * <pre>
 * get_local {var}
 * {operand}
 * {op}
 * set_local {var}
 * </pre>
 */
wabt::ExprList* GenerateCompoundAssignment(wabt::Var var, wabt::Opcode op, wabt::ExprList* operand);

/*!
 * Generate horizontal sum
 * <pre>
 * f32.extract_lane {var} 0
 * f32.extract_lane {var} 1
 * f32.add
 * f32.extract_lane {var} 2
 * f32.add
 * f32.extract_lane {var} 3
 * f32.add
 * </pre>
 * @param var Reference variable
 * @return Expression list
 */
  wabt::ExprList* GenerateF32X4HorizontalLTRSum(wabt::Var var);

} // namespace wasmpp

#endif