#ifndef NN_SNIPPETS_SNIPPETS_H_
#define NN_SNIPPETS_SNIPPETS_H_

#include <string>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace arch {
  class BuiltinFunctions;
}
namespace snippet {

class Snippet {
protected:
  wasmpp::LabelManager* label_manager_ = nullptr;
  arch::BuiltinFunctions* builtins_ = nullptr;
public:
  Snippet(wasmpp::LabelManager* label_manager, arch::BuiltinFunctions* builtins) :
      label_manager_(label_manager), builtins_(builtins) {}
};

} // namespace builtins
} // namespace nn

#endif
