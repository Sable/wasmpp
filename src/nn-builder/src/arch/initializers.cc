#include <src/nn-builder/src/arch/initializers.h>
#include <random>
#include <cmath>

namespace nn {
namespace arch {

std::vector<wasmpp::DataEntry> XavierDistribution(uint32_t size, uint32_t n_in, uint32_t n_out, bool uniform, uint64_t seed) {
  if(uniform) {
    float limit = sqrtf(6.0f / (n_in + n_out));
    return UniformDistribution(size, -limit, limit, seed);
  }

  float mean = 0;
  float std_dev = sqrtf(2.0f / (n_in + n_out));
  return GaussianDistribution(size, mean, std_dev, seed);
}

std::vector<wasmpp::DataEntry> LeCunDistribution(uint32_t size, uint32_t n_in, bool uniform, uint64_t seed) {
  if(uniform) {
    float limit = sqrtf(3.0f / n_in);
    return UniformDistribution(size, -limit, limit, seed);
  }

  float mean = 0;
  float std_dev = sqrtf(1.0f / n_in);
  return GaussianDistribution(size, mean, std_dev, seed);
}

std::vector<wasmpp::DataEntry> GaussianDistribution(uint32_t size, float mean, float std_dev, uint64_t seed) {
  std::default_random_engine generator(seed);
  std::vector<wasmpp::DataEntry> entries;
  std::normal_distribution<float> distribution(mean, std_dev);
  while (size-- > 0) {
    entries.push_back(wasmpp::DataEntry::MakeF32(distribution(generator)));
  }
  return entries;
}

std::vector<wasmpp::DataEntry> UniformDistribution(uint32_t size, float low, float high, uint64_t seed) {
  std::default_random_engine generator(seed);
  std::vector<wasmpp::DataEntry> entries;
  std::uniform_real_distribution<float> distribution(low, high);
  while(size-- > 0) {
    entries.push_back(wasmpp::DataEntry::MakeF32(distribution(generator)));
  }
  return entries;
}

std::vector<wasmpp::DataEntry> ConstantDistribution(uint32_t size, float value) {
  std::vector<wasmpp::DataEntry> entries;
  while(size-- > 0) {
    entries.push_back(wasmpp::DataEntry::MakeF32(value));
  }
  return entries;
}

} // namespace arch
} // namespace nn
