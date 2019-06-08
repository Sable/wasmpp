#ifndef NN_SNIPPET_CONFUSION_MATRIX_H_
#define NN_SNIPPET_CONFUSION_MATRIX_H_

#include <src/nn-builder/src/snippet/matrix.h>

namespace nn {
namespace snippet {

class ConfusionMatrixSnippet : public Snippet {
public:
  ConfusionMatrixSnippet(wasmpp::LabelManager* label_manager) : Snippet(label_manager) {}

  // Update confusion matrix
  virtual wabt::ExprList* ConfusionMatrixUpdate(ds::NDArray* matrix, ds::NDArray* predictions, RelocMat target,
                                                std::vector<wabt::Var> locals);
};

class ConfusionMatrixSnippetSimd : public ConfusionMatrixSnippet {
public:
  ConfusionMatrixSnippetSimd(wasmpp::LabelManager* label_manager) : ConfusionMatrixSnippet(label_manager) {}
};

} // namespace snippet
} // namespace nn

#endif
