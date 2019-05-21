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

  const Type type = Type::F64;
  uint64_t type_size = TypeSize(type);
  for(auto l = 0; l < layers_.size(); ++l) {
    auto layer = CAST(layers_[l], FullyConnectedLayer);
    // values
    auto memZ = module_manager_.Memory().Allocate(layer->Nodes() * type_size);
    Z[l] = new ds::NDArray(memZ, {layer->Nodes(), 1}, type_size);
    if(l > 0) {
      auto prev_layer = CAST(layers_[l-1], FullyConnectedLayer);
      // weight
      auto memW = module_manager_.Memory().Allocate(layer->Nodes() * prev_layer->Nodes() * type_size);
      W[l] = new ds::NDArray(memW, {layer->Nodes(), prev_layer->Nodes()}, type_size);
      // bias
      auto memb = module_manager_.Memory().Allocate(layer->Nodes() * type_size);
      b[l] = new ds::NDArray(memb, {layer->Nodes(), 1}, type_size);
    }
  }

  module_manager_.MakeMemory(1);
  auto train = module_manager_.MakeFunction("main", {}, {Type::I32, Type::I32},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    uint64_t side = 3;
    auto a1 = module_manager_.Memory().Allocate(side * side * type_size);
    ds::NDArray A1(a1, {side, side}, type_size);

    auto a2 = module_manager_.Memory().Allocate(side * side * type_size);
    ds::NDArray A2(a2, {side, side}, type_size);

    auto a3 = module_manager_.Memory().Allocate(side * side * type_size);
    ds::NDArray A3(a3, {side, side}, type_size);

    for(int r=0; r < side; r++) {
      for(int c=0; c < side; c++) {
        f.Insert(MakeF64Store(MakeI32Const(A1.GetLinearIndex({r, c})), MakeF64Const(side)));
        f.Insert(MakeF64Store(MakeI32Const(A2.GetLinearIndex({r, c})), MakeF64Const(side)));
      }
    }
    f.Insert(math::Add2DArrays<type>(f.Label(), A1, A2, A3, locals));

    for(int r=0; r < side; r++) {
      for(int c=0; c < side; c++) {
        f.Insert(MakeCall(builtins.print_f64, {MakeF64Load(MakeI32Const(A3.GetLinearIndex({r, c})))}));
      }
    }
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
