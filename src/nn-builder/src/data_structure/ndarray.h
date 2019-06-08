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
  std::vector<uint32_t> shape_;
  uint32_t unit_size_;
  // Store the multiplication value for the shape elements
  // left to right. This is useful for computing the index
  // in the linear memory
  std::vector<uint32_t> shape_mul_;
public:
  NDArray(wasmpp::Memory* memory, std::vector<uint32_t> shape, uint32_t unit_size);
  void Reshape(std::vector<uint32_t> shape);
  std::vector<uint32_t >Shape() const { return shape_;}
  uint32_t GetLinearIndex(std::vector<uint32_t> index) const;
  const wasmpp::Memory* Memory() const { return memory_; }
  uint32_t Begin() const;
  uint32_t End() const;
};

} // namespace ds
} // namespace nn

#endif
