#include <src/wasmpp/common.h>

namespace wasmpp {

uint32_t TypeSize(wabt::Type type) {
  switch (type) {
    case wabt::Type::I32:
      return WASMPP_I32_SIZE;
    case wabt::Type::F32:
      return WASMPP_F32_SIZE;
    case wabt::Type::I64:
      return WASMPP_I64_SIZE;
    case wabt::Type::F64:
      return WASMPP_F64_SIZE;
    case wabt::Type::V128:
      return WASMPP_V128_SIZE;
  }
  assert(!"type not supported");
}

} // namespace wasmpp