#ifndef NN_SNIPPETS_SNIPPETS_H_
#define NN_SNIPPETS_SNIPPETS_H_

#include <string>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace snippet {

class Snippet {
protected:
  wasmpp::LabelManager* label_manager_ = nullptr;
public:
  Snippet(wasmpp::LabelManager* label_manager) : label_manager_(label_manager) {}
};

} // namespace builtins
} // namespace nn

#endif
