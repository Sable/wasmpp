#ifndef NN_BUILTINS_SYSTEM_H_
#define NN_BUILTINS_SYSTEM_H_

#include <src/apps/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

class System : public Builtin {
private:
  wabt::Var print_i32_;
  wabt::Var print_i64_;
  wabt::Var print_f32_;
  wabt::Var print_f64_;
  wabt::Var print_table_f64_;
public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) override;
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) override;

  const wabt::Var& PrintI32() const { return print_i32_; }
  const wabt::Var& PrintI64() const { return print_i64_; }
  const wabt::Var& PrintF32() const { return print_f32_; }
  const wabt::Var& PrintF64() const { return print_f64_; }
  const wabt::Var& PrintTableF64() const { return print_table_f64_; }
};

} // namespace builtins
} // namespace nn

#endif
