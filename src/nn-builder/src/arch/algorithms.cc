#include <src/nn-builder/src/arch/algorithms.h>
#include <random>
#include <cmath>

namespace nn {
namespace arch {

std::vector<wasmpp::DataEntry> Xavier(uint32_t size, uint32_t n_in, uint32_t n_out, bool uniform) {
  std::vector<wasmpp::DataEntry> entries;
  std::default_random_engine generator;
  if(uniform) {
    float limit = sqrtf(6.0f / (n_in + n_out));
    std::uniform_real_distribution<float> distribution(-limit, limit);
    while(size-- > 0) {
      entries.push_back(wasmpp::DataEntry::MakeF32(distribution(generator)));
    }
  } else {
    int mean = 0;
    float std_dev = sqrtf(2.0f / (n_in + n_out));
    std::normal_distribution<float> distribution(mean, std_dev);
    // TODO Check if need to add condition for number boundaries
    while(size-- > 0) {
      entries.push_back(wasmpp::DataEntry::MakeF32(distribution(generator)));
    }
  }
  return entries;
}

std::vector<wasmpp::DataEntry> LeCun(uint32_t size, uint32_t n_in, bool uniform) {
  std::vector<wasmpp::DataEntry> entries;
  std::default_random_engine generator;
  if(uniform) {
    float limit = sqrtf(3.0f / n_in);
    std::uniform_real_distribution<float> distribution(-limit, limit);
    while(size-- > 0) {
      entries.push_back(wasmpp::DataEntry::MakeF32(distribution(generator)));
    }
  } else {
    int mean = 0;
    float std_dev = sqrtf(1.0f / n_in);
    std::normal_distribution<float> distribution(mean, std_dev);
    // TODO Check if need to add condition for number boundaries
    while (size-- > 0) {
      entries.push_back(wasmpp::DataEntry::MakeF32(distribution(generator)));
    }
  }
  return entries;
}

} // namespace arch
} // namespace nn
