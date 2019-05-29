#include <src/nn-builder/src/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>
#include <cassert>

namespace nn {
namespace ds {

NDArray::NDArray(wasmpp::Memory *memory, std::vector<uint64_t> shape, uint64_t unit_size) {
  ERROR_UNLESS(memory != nullptr, "memory cannot be null");
  ERROR_UNLESS(unit_size > 0, "unit size cannot be null");
  ERROR_UNLESS(unit_size <= memory->Bytes(), "unit size cannot be greater than the number of bytes");
  ERROR_UNLESS(memory->Bytes() % unit_size == 0, "number of bytes must be a multiple of unit size");
  memory_ = memory;
  unit_size_ = unit_size;
  Reshape(shape);
}

uint64_t NDArray::GetLinearIndex(std::vector<uint64_t> index) const {
  ERROR_UNLESS(index.size() == shape_.size(), "wrong index shape");
  uint64_t lindex = memory_->Begin();
  for(int i = 0; i < index.size(); i++) {
    ERROR_UNLESS(index[i] < shape_[i], "index out of bound in shape at position %d", i);
    lindex += index[i] * shape_mul_[i];
  }
  assert(lindex >= memory_->Begin());
  assert(lindex < memory_->End());
  return lindex;
}

void NDArray::Reshape(std::vector<uint64_t> shape) {
  ERROR_UNLESS(!shape.empty(), "shape cannot be empty");
  ERROR_UNLESS(memory_ != nullptr, "memory cannot be null");
  uint64_t total = unit_size_;
  for(auto val : shape) {
    total *= val;
  }
  ERROR_UNLESS(total == memory_->Bytes(), "new shape is not compatible with the amount of bytes");

  shape_ = shape;
  // Optimize the index computation
  shape_mul_.resize(shape_.size());
  shape_mul_[shape_.size() - 1] = unit_size_;
  for(size_t i=shape_.size()-1; i > 0; i--) {
    shape_mul_[i-1] = shape_[i] * shape_mul_[i];
  }
}

} // namespace ds
} // namespace nn