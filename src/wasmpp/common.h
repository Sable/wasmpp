/*!
 * @file common.h
 */
#ifndef WASMPP_COMMON_H_
#define WASMPP_COMMON_H_

#include <src/common.h>
#include <src/opcode.h>

namespace wasmpp {

#define ERROR_EXIT(...)                           \
do {                                              \
  fprintf(stderr, "%s:%d\n", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__);                   \
  fprintf(stderr, "\n");                          \
  exit(1);                                        \
} while(0);

#define ERROR_UNLESS(cond, ...) \
do {                            \
  if(!(cond)) {                 \
    ERROR_EXIT(__VA_ARGS__);    \
  }                             \
} while(0);

// Wasm type sizes in bytes
const uint32_t WASMPP_I32_SIZE = 4;
const uint32_t WASMPP_F32_SIZE = 4;
const uint32_t WASMPP_I64_SIZE = 8;
const uint32_t WASMPP_F64_SIZE = 8;
const uint32_t WASMPP_V128_SIZE = 16;

// Size of types
/*!
 * Get Wasm type size
 * @param type Wasm type
 * @return size in bytes
 */
uint32_t TypeSize(wabt::Type type);

/*!
 * Shift left value for a Wasm type <br/>
 * e.g. <code>sizeof(i32) = 4</code> -> <code>(1 << 2)</code>. <br/>
 * Therefore <code>TypeShiftLeft(i32)</code> is 2
 * @param type
 * @return
 */
uint32_t TypeShiftLeft(wabt::Type type);

/*!
 * Get the SIMD version of a Wasm instruction
 * @param op Opcode of the non-SIMD instruction
 * @return Opcode of the SIMD instruction
 */
wabt::Opcode OpcodeToSimdOpcode(wabt::Opcode op);

} // namespace wasmpp

#endif
