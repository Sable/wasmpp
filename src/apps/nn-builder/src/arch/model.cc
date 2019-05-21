#include <src/apps/nn-builder/src/arch/model.h>
#include <src/apps/nn-builder/src/data_structure/ndarray.h>
#include <src/apps/nn-builder/src/math/matrix.h>

namespace nn {
namespace arch {

using namespace wasmpp;
using namespace wabt;

void Model::AddLayer(layer_sptr layer) {
  layers_.push_back(layer);
}

bool Model::RemoveLayer(uint32_t index) {
  if(index >= layers_.size()) {
    return false;
  }
  layers_.erase(layers_.begin() + index);
  return true;
}

layer_sptr Model::GetLayer(uint32_t index) const {
  if(index < layers_.size()) {
    return layers_[index];
  }
  assert(!"Index out of bound");
}

void Model::InitImports() {
  builtins.print_i32 = module_manager_.MakeFuncImport("System", "print", {{Type::I32}, {}});
  builtins.print_i64 = module_manager_.MakeFuncImport("System", "print", {{Type::I64}, {}});
  builtins.print_f32 = module_manager_.MakeFuncImport("System", "print", {{Type::F32}, {}});
  builtins.print_f64 = module_manager_.MakeFuncImport("System", "print", {{Type::F64}, {}});
  builtins.exp = module_manager_.MakeFuncImport("Math", "exp", {{Type::F64}, {Type::F64}});
}

void Model::InitDefinitions() {
  builtins.sigmoid = module_manager_.MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto denom = MakeBinary(Opcode::F64Add, MakeF64Const(1),
        MakeCall(builtins.exp, {MakeUnary(Opcode::F64Neg, MakeLocalGet(params[0]))}));
    auto div = MakeBinary(Opcode::F64Div, MakeF64Const(1), denom);
    f.Insert(div);
  });
}

void Model::Setup() {
  InitImports();
  InitDefinitions();

#define CAST(x, cls) std::static_pointer_cast<cls>(x);

  std::vector<ds::NDArray*> Z(layers_.size());
  std::vector<ds::NDArray*> W(layers_.size());
  std::vector<ds::NDArray*> b(layers_.size());

  const Type val_type = Type::F64;
  const uint64_t val_type_size = TypeSize(val_type);
  for(auto l = 0; l < layers_.size(); ++l) {
    auto layer = CAST(layers_[l], FullyConnectedLayer);
    // values
    auto memZ = module_manager_.Memory().Allocate(layer->Nodes() * val_type_size);
    Z[l] = new ds::NDArray(memZ, {layer->Nodes(), 1}, val_type_size);
    if(l > 0) {
      auto prev_layer = CAST(layers_[l-1], FullyConnectedLayer);
      // weight
      auto memW = module_manager_.Memory().Allocate(layer->Nodes() * prev_layer->Nodes() * val_type_size);
      W[l] = new ds::NDArray(memW, {layer->Nodes(), prev_layer->Nodes()}, val_type_size);
      // bias
      auto memb = module_manager_.Memory().Allocate(layer->Nodes() * val_type_size);
      b[l] = new ds::NDArray(memb, {layer->Nodes(), 1}, val_type_size);
    }
  }

  module_manager_.MakeMemory(module_manager_.Memory().Pages());
  auto feedforward = module_manager_.MakeFunction("feedforward", {},
      {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, val_type},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vi32_6 = locals[5];
    auto vtype_1 = locals[6];
    for(int l=1; l < layers_.size(); ++l) {
      auto mul = math::Multiply2DArrays<val_type>(f.Label(), *W[l], *Z[l-1], *Z[l],
          {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vi32_6, vtype_1});
      auto add = math::Add2DArrays<val_type>(f.Label(), *Z[l], *b[l], *Z[l], {vi32_1, vi32_2});
      auto app = math::ApplyFx2DArrays<val_type>(f.Label(), *Z[l], builtins.sigmoid, *Z[l], {vi32_1, vi32_2});
      f.Insert(mul);
      f.Insert(add);
      f.Insert(app);
    }
  });

  auto train = module_manager_.MakeFunction("train", {}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeCall(feedforward, {}));
  });

  auto main = module_manager_.MakeFunction("main", {}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeCall(train, {}));
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
