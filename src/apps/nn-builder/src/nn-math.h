#ifndef NN_BUILDER_MATH_H
#define NN_BUILDER_MATH_H

#include <src/wasmpp/wasm-builder.h>
#include <src/ir.h>
#include <map>

namespace nn {

class MathExtension {
private:
  wasmpp::ModuleBuilder* mb_ = nullptr;

  // Functions vars
  std::map<std::string, wabt::Var> func_map;

public:
  MathExtension(wasmpp::ModuleBuilder* mb) : mb_(mb) {}

  // Sigmoid function
  wabt::Var Sigmoid();

  // Exp function
  wabt::Var Exp();
};

} // namespace nn

#endif