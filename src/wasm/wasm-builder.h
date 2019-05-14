#ifndef WASM_WASM_BUILDER_H_
#define WASM_WASM_BUILDER_H_

#include <src/ir.h>

namespace wasm {

class ModuleBuilder {
private:
  wabt::Module module_;
public:

  const wabt::Module& GetModule() const;

  // Create a new function in a module
  wabt::Func* CreateFunction();

};

} // namespace wasm

#endif
