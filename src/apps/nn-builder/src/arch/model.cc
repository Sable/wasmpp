#include <src/apps/nn-builder/src/arch/model.h>
#include <src/apps/nn-builder/src/data_structure/ndarray.h>

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

  std::vector<ds::NDArray*> vals;
  std::vector<ds::NDArray*> weights;
  for(auto i = 0;i < layers_.size(); ++i) {
    uint32_t type_size = TypeSize(Type::F64);
    auto fc_layer = std::static_pointer_cast<FullyConnectedLayer>(layers_[i]);
    auto layer_vals_mem = module_manager_.Memory().Allocate(fc_layer->Nodes() * type_size);
    vals.push_back(new ds::NDArray(layer_vals_mem, {fc_layer->Nodes(), 1}, type_size));
    printf("(%ld, %ld)\n", vals.back()->Shape()[0], vals.back()->Shape()[1]);
  }

  auto train = module_manager_.MakeFunction("main", {}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeCall(builtins.print_i32, {MakeI32Const(2)}));
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
