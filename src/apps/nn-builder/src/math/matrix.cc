#include <src/apps/nn-builder/src/math/matrix.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace nn {
namespace compute {
namespace math {

using namespace wasmpp;
using namespace wabt;

void Multiply2DArrays(ModuleManager* mm, ContentManager* ctn,
                      NDArray lhs, NDArray rhs, NDArray dst, Var x, Var y, Var z) {
  assert(lhs.Shape().size() == 2);
  assert(rhs.Shape().size() == 2);
  assert(dst.Shape().size() == 2);
  auto loopX = GenerateRangeLoop(mm, x, 0, 100, 1, [&](BlockBody* bX) {
    auto loopY = GenerateRangeLoop(mm, y, 0, 100, 1, [&](BlockBody* bY) {
      auto loopZ = GenerateRangeLoop(mm, z, 0, 100, 1, [&](BlockBody* bZ) {

      });
      bY->Insert(loopZ);
    });
    bX->Insert(loopY);
  });
  ctn->Insert(loopX);
}

} // namespace math
} // namespace compute
} // namespace nn
