#ifndef WASMPP_COMMON_H_
#define WASMPP_COMMON_H_

#include <src/common.h>

namespace wasmpp {

#define ERROR_EXIT(...) \
fprintf(stderr, __VA_ARGS__); \
exit(1);

// Size of types
uint32_t TypeSize(wabt::Type type);

}

#endif
