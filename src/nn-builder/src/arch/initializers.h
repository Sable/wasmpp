#ifndef NN_ARCH_INITIALIZERS_H_
#define NN_ARCH_INITIALIZERS_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/nn-builder/src/data_structure/ndarray.h>

namespace nn {
namespace arch {

enum WeightDistributionType {
  // Distribution that can be applied
  // only to the hidden layers and not
  // the output layer. If a new function
  // is added make sure to add it here and
  // update the FIRST and LAST
  XavierUniform,
  XavierNormal,
  FIRST_HIDDEN_ONLY = XavierUniform,
  LAST_HIDDEN_ONLY = XavierNormal,

  // Distribution that can be applied
  // to the hidden layers and
  // output layer. If a new function
  // is added make sure to add it here and
  // update the FIRST and LAST
  LeCunUniform,
  LeCunNormal,
  Uniform,
  Gaussian,
  Constant,
  FIRST_HIDDEN_OR_OUTPUT = LeCunUniform,
  LAST_HIDDEN_OR_OUTPUT = Constant

};

struct WeightDistributionOptions {
  float gaussian_mean = 0;
  float gaussian_std_dev = 0.1;
  float uniform_low = -1;
  float uniform_high = 1;
  float constant_value = 0.1;
  uint64_t seed = 0;
};

std::vector<wasmpp::DataEntry> XavierDistribution(uint32_t size, uint32_t n_in, uint32_t n_out, bool uniform, uint64_t seed);
std::vector<wasmpp::DataEntry> LeCunDistribution(uint32_t size, uint32_t n_in, bool uniform, uint64_t seed);
std::vector<wasmpp::DataEntry> GaussianDistribution(uint32_t size, float mean, float std_dev, uint64_t seed);
std::vector<wasmpp::DataEntry> UniformDistribution(uint32_t size, float low, float high, uint64_t seed);
std::vector<wasmpp::DataEntry> ConstantDistribution(uint32_t size, float value);

} // namespace arch
} // namespace nn

#endif
