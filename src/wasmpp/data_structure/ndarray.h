#ifndef WASM_WASM_NDARRAY_H_
#define WASM_WASM_NDARRAY_H_

#include <vector>
#include <stdint.h>

namespace wasmpp {
class Memory;

class NDArray {
private:
  Memory* memory_;
  std::vector<uint64_t> shape_;
  // Store the multiplication value for the shape elements
  // left to right. This is useful for computing the index
  // in the linear memory
  std::vector<uint64_t> shape_mul_;
public:
  NDArray(Memory* memory, std::vector<uint64_t> shape);
  void Reshape(std::vector<uint64_t> shape);
  std::vector<uint64_t >GetShape() const { return shape_;}
  uint64_t GetLinearIndex(std::vector<uint64_t> index) const;
};

} // namespace wasmpp

#endif
