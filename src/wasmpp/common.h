#ifndef WASMPP_COMMON_H_
#define WASMPP_COMMON_H_

#include <src/common.h>

namespace wasmpp {

#define ERROR_EXIT(...)         \
do {                            \
  fprintf(stderr, __VA_ARGS__); \
  exit(1);                      \
} while(0);

#define ERROR_UNLESS(cond, ...) \
do {                            \
  if(!(cond)) {                 \
    ERROR_EXIT(__VA_ARGS__);    \
  }                             \
} while(0);

// Wasm type sizes in bytes
const uint32_t WASMPP_I32_SIZE = 4;
const uint32_t WASMPP_I64_SIZE = 4;
const uint32_t WASMPP_F32_SIZE = 8;
const uint32_t WASMPP_F64_SIZE = 8;
const uint32_t WASMPP_V128_SIZE = 16;

// Size of types
uint32_t TypeSize(wabt::Type type);

}

#endif
