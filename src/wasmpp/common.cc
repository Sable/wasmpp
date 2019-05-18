#include <src/wasmpp/common.h>

namespace wasmpp {

uint32_t TypeSize(wabt::Type type) {
  switch (type) {
    case wabt::Type::I32:
    case wabt::Type::F32:
      return 4;
    case wabt::Type::I64:
    case wabt::Type::F64:
      return 8;
    case wabt::Type::V128:
      return 16;
  }
  assert(!"type not supported");
}

} // namespace wasmpp