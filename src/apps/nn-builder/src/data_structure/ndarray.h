#ifndef NN_DS_NDARRAY_H_
#define NN_DS_NDARRAY_H_

#include <vector>
#include <cstdint>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace ds {

class NDArray {
private:
  wasmpp::Memory* memory_;
  std::vector<uint64_t> shape_;
  uint64_t unit_size_;
  // Store the multiplication value for the shape elements
  // left to right. This is useful for computing the index
  // in the linear memory
  std::vector<uint64_t> shape_mul_;
public:
  NDArray(wasmpp::Memory* memory, std::vector<uint64_t> shape, uint64_t unit_size);
  void Reshape(std::vector<uint64_t> shape);
  std::vector<uint64_t >Shape() const { return shape_;}
  uint64_t GetLinearIndex(std::vector<uint64_t> index) const;
};

} // namespace ds
} // namespace nn

#endif
