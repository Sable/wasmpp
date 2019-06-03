#ifndef NN_ARCH_ALGORITHMS_H_
#define NN_ARCH_ALGORITHMS_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/nn-builder/src/data_structure/ndarray.h>

namespace nn {
namespace arch {

std::vector<wasmpp::DataEntry> Xavier(uint32_t size, uint32_t n_in, uint32_t n_out, bool uniform);

std::vector<wasmpp::DataEntry> LeCun(uint32_t size, uint32_t n_in, bool uniform);

} // namespace arch
} // namespace nn

#endif
