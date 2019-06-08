#ifndef NN_SNIPPET_ANALYSIS_H_
#define NN_SNIPPET_ANALYSIS_H_

#include <src/nn-builder/src/snippet/matrix.h>

namespace nn {
namespace snippet {

class AnalysisSnippet : public Snippet {
public:
  AnalysisSnippet(wasmpp::LabelManager* label_manager) : Snippet(label_manager) {}

  // Update confusion matrix
  // This function expects the prediction to be hard-maxed already
  virtual wabt::ExprList* ConfusionMatrixUpdate(ds::NDArray* matrix, ds::NDArray* predictions, RelocMat target,
                                                std::vector<wabt::Var> locals);

  // Compute correct predictions
  // This function expects the prediction to be hard-maxed already
  virtual wabt::ExprList* CorrectPredictions(ds::NDArray* predictions, RelocMat target, wabt::Var correct_predictions,
                                             std::vector<wabt::Var> locals);
};

class AnalysisSnippetSimd : public AnalysisSnippet {
public:
  AnalysisSnippetSimd(wasmpp::LabelManager* label_manager) : AnalysisSnippet(label_manager) {}
};

} // namespace snippet
} // namespace nn

#endif
