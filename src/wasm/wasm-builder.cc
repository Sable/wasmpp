#include <src/wasm/wasm-builder.h>

namespace wasm {

const wabt::Module& ModuleBuilder::GetModule() const {
  return module_;
}

wabt::Func* ModuleBuilder::CreateFunction() {
  auto field = wabt::MakeUnique<wabt::FuncModuleField>();
  module_.AppendField(std::move(field));
  return module_.funcs.back();
}

} // namespace wasm

