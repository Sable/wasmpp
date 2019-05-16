#include <src/wasmpp/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>
#include <cassert>

namespace wasmpp {

NDArray::NDArray(wasmpp::Memory *memory, std::vector<uint64_t> shape) {
  memory_ = memory;
  Reshape(shape);
}

uint64_t NDArray::GetLinearIndex(std::vector<uint64_t> index) const {
  assert(index.size() == shape_.size());
  uint64_t lindex = memory_->begin;
  for(int i = 0; i < index.size(); i++) {
    assert(index[i] >= 0);
    assert(index[i] < shape_[i]);
    lindex += index[i] * shape_mul_[i];
  }
  assert(lindex >= memory_->begin);
  assert(lindex < memory_->begin + memory_->Size());
  return lindex;
}

void NDArray::Reshape(std::vector<uint64_t> shape) {
  assert(!shape.empty());
  assert(memory_ != nullptr);
  uint64_t total = 1;
  for(auto val : shape) {
    total *= val;
  }
  assert(total == memory_->Size());

  shape_ = shape;
  // Optimize the index computation
  shape_mul_.resize(shape_.size());
  shape_mul_[shape_.size() - 1] = 1;
  for(size_t i=shape_.size()-1; i > 0; i--) {
    shape_mul_[i-1] = shape_[i] * shape_mul_[i];
  }
}

} // namespace wasmpp