#include <src/nn-builder/src/snippet/analysis.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace nn {
namespace snippet {

using namespace wasmpp;
using namespace wabt;
using namespace ds;

wabt::ExprList* AnalysisSnippet::ConfusionMatrixUpdate(nn::ds::NDArray *matrix, nn::ds::NDArray *predictions,
                                                       RelocMat target, std::vector<wabt::Var> locals) {
  MATRIX_CHECK(matrix);
  MATRIX_CHECK(predictions);
  MATRIX_CHECK(target.Array());
  MATRIX_SAME_SHAPE(predictions, target.Array());
  ERROR_UNLESS(matrix->Shape()[0] == matrix->Shape()[1], "confusion matrix must be a square");
  ERROR_UNLESS(matrix->Shape()[0] == predictions->Shape()[0], "confusion matrix and predictions are not compatible");
  assert(locals.size() == 6);

  auto col = locals[0];
  auto row = locals[1];
  auto rel_row = locals[2];
  auto x = locals[3];
  auto y = locals[4];
  auto offset = locals[5];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t width = predictions->Shape()[1] * type_size;
  uint32_t height = predictions->Shape()[0] * width;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, GenerateRangeLoop(label_manager_, col, 0, width, type_size, {}, [&](BlockBody* b1) {
    // Find 1 in both A and target
    b1->Insert(MakeLocalSet(rel_row, MakeI32Const(0)));
    b1->Insert(GenerateRangeLoop(label_manager_, row, 0, height, width, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(offset, MakeBinary(Opcode::I32Add, MakeLocalGet(col), MakeLocalGet(row))));
      wabt::ExprList* target_curr = nullptr;
      if(target.HasBeginVar()) {
         target_curr = MakeBinary(Opcode::I32Add, MakeLocalGet(offset), MakeLocalGet(target.Var()));
      } else {
        target_curr = MakeBinary(Opcode::I32Add, MakeLocalGet(offset), MakeI32Const(target.Array()->Begin()));
      }
      auto pred_curr = MakeBinary(Opcode::I32Add, MakeLocalGet(offset), MakeI32Const(predictions->Begin()));
      // If 1 found in target, store it in y
      b2->Insert(MakeIf(label_manager_, MakeBinary(Opcode::F32Eq, MakeF32Load(target_curr), MakeF32Const(1)), {},
                        [&](BlockBody t, Var label) {
                          t.Insert(MakeLocalSet(y, MakeLocalGet(rel_row)));
                        }));
      // If 1 found in prediction, store it in x
      b2->Insert(MakeIf(label_manager_, MakeBinary(Opcode::F32Eq, MakeF32Load(pred_curr), MakeF32Const(1)), {},
                        [&](BlockBody t, Var label) {
                          t.Insert(MakeLocalSet(x, MakeLocalGet(rel_row)));
                        }));
      b2->Insert(GenerateCompoundAssignment(rel_row, Opcode::I32Add, MakeI32Const(type_size)));
    }));

    // Add 1 in confusion matrix
    auto cm_y = MakeBinary(Opcode::I32Mul, MakeLocalGet(y), MakeI32Const(matrix->Shape()[0]));
    b1->Insert(MakeLocalSet(offset, MakeBinary(Opcode::I32Add, MakeLocalGet(x), cm_y)));
    b1->Insert(GenerateCompoundAssignment(offset, Opcode::I32Add, MakeI32Const(matrix->Memory()->Begin())));
    b1->Insert(MakeF32Store(MakeLocalGet(offset), MakeBinary(Opcode::F32Add, MakeF32Load(MakeLocalGet(offset)),
                                                             MakeF32Const(1))));
  }));
  return e;
}

wabt::ExprList* AnalysisSnippet::CorrectPredictions(nn::ds::NDArray *predictions, nn::snippet::RelocMat target,
                                                    wabt::Var correct_predictions, std::vector<wabt::Var> locals) {
  MATRIX_CHECK(predictions);
  VECTOR_CHECK(target.Array());
  MATRIX_SAME_SHAPE(predictions, target.Array());

  assert(locals.size() == 3);

  auto row = locals[0];
  auto col = locals[1];
  auto addr = locals[2];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t width_bytes = predictions->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, MakeLocalSet(correct_predictions, MakeF32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, row, 0, predictions->Memory()->Bytes(), width_bytes, {}, [&](BlockBody* b1) {
    b1->Insert(GenerateRangeLoop(label_manager_, col, 0, width_bytes, type_size, {}, [&](BlockBody* b2){
      auto pre_addr = MakeBinary(Opcode::I32Add, MakeI32Const(predictions->Begin()), MakeLocalGet(addr));
      b2->Insert(MakeIf(label_manager_, MakeBinary(Opcode::F32Eq, MakeF32Load(pre_addr), MakeF32Const(1.0f)), {},
                        [&](BlockBody t1, Var label){
        ExprList* tar_addr = nullptr;
        if(target.HasBeginVar()) {
          tar_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(target.Var()), MakeLocalGet(addr));
        } else {
          tar_addr = MakeBinary(Opcode::I32Add, MakeI32Const(target.Array()->Begin()), MakeLocalGet(addr));
        }
        t1.Insert(MakeIf(label_manager_, MakeBinary(Opcode::F32Eq, MakeF32Load(tar_addr), MakeF32Const(1.0f)), {},
                        [&](BlockBody t2, Var){
          t2.Insert(GenerateCompoundAssignment(correct_predictions, Opcode::F32Add, MakeF32Const(1.0f)));
        }));
      }));
      b2->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
    }));
  }));
  return e;}

} // namespace snippet
} // namespace nn
