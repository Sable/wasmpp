#include <src/nn-builder/src/builtins/math.h>
#include <src/wasmpp/wasm-instructions-gen.h>
#include <cassert>

namespace nn {
namespace builtins {

using namespace wabt;
using namespace wasmpp;

void Math::InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
  exp_ = module_manager->MakeFuncImport(module_name, "exp", {{Type::F32}, {Type::F32}});
  log_ = module_manager->MakeFuncImport(module_name, "log", {{Type::F32}, {Type::F32}});
  random_ = module_manager->MakeFuncImport(module_name, "random", {{}, {Type::F32}});
}

void Math::InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) {
  assert(model != nullptr);
  assert(module_manager != nullptr);

  mask_matrix_ = module_manager->MakeFunction(nullptr, {{Type::I32, Type::I32, Type::F32}, {}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto begin = params[0];
    auto end = params[1];
    auto keep_prob = params[2];
    f.Insert(GenerateDoWhileLoop(f.Label(), begin, end, TypeSize(Type::F64), {}, [&](BlockBody *b){
      b->Insert(MakeIf(f.Label(), MakeBinary(Opcode::F32Ge, MakeLocalGet(keep_prob), MakeCall(Random(), {})), {},
                       [&](BlockBody true_block, Var true_label){
        true_block.Insert(MakeF32Store(MakeLocalGet(begin), MakeF32Const(1)));
      }, [&](BlockBody false_block){
        false_block.Insert(MakeF32Store(MakeLocalGet(begin), MakeF32Const(0)));
      }));
    }));
  });
}

} // namespace builtins
} // namespace nn
