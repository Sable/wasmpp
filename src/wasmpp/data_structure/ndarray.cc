#include <src/wasmpp/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>
#include <cassert>

namespace wasmpp {

NDArray::NDArray(wasmpp::Memory *memory, std::vector<uint64_t> shape, uint64_t unit_size) {
  assert(memory != nullptr);
  assert(unit_size > 0);
  assert(unit_size <= memory->Bytes());
  assert(memory->Bytes() % unit_size == 0);
  memory_ = memory;
  unit_size_ = unit_size;
  Reshape(shape);
}

uint64_t NDArray::GetLinearIndex(std::vector<uint64_t> index) const {
  assert(index.size() == shape_.size());
  uint64_t lindex = memory_->Begin();
  for(int i = 0; i < index.size(); i++) {
    assert(index[i] >= 0);
    assert(index[i] < shape_[i]);
    lindex += index[i] * shape_mul_[i];
  }
  assert(lindex >= memory_->Begin());
  assert(lindex < memory_->End());
  return lindex;
}

void NDArray::Reshape(std::vector<uint64_t> shape) {
  assert(!shape.empty());
  assert(memory_ != nullptr);
  uint64_t total = unit_size_;
  for(auto val : shape) {
    total *= val;
  }
  assert(total == memory_->Bytes());

  shape_ = shape;
  // Optimize the index computation
  shape_mul_.resize(shape_.size());
  shape_mul_[shape_.size() - 1] = unit_size_;
  for(size_t i=shape_.size()-1; i > 0; i--) {
    shape_mul_[i-1] = shape_[i] * shape_mul_[i];
  }
}

} // namespace wasmpp