#include <src/wasmpp/builtins/memory-builtins.h>
#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions-gen.h>
#include <src/wasmpp/common.h>

namespace wasmpp {
using namespace wabt;

void MemoryBuiltins::InitImports() {}

void MemoryBuiltins::InitDefinitions() {
  if(options_->memory.fill_i32)
    fill_i32_ = BuildFillI32();
  if(options_->memory.fill_i64)
    fill_i64_ = BuildFillI64();
  if(options_->memory.fill_f32)
    fill_f32_ = BuildFillF32();
  if(options_->memory.fill_f64)
    fill_f64_ = BuildFillF64();
}

template <Type type>
exprs_sptr Fill(ContentManager* ctn, const Var &offset, const Var &repeat, const Var &value) {
  auto e = CreateExprList();
  uint32_t type_size = TypeSize(type);
  auto loop = MakeLoop(ctn, [&](BlockBody b, Var label) {
    auto store_func = MakeI32Store;
    switch (type) {
      case Type::I32:
        // do nothing
        break;
      case Type::I64:
        store_func = MakeI64Store;
        break;
      case Type::F32:
        store_func = MakeF32Store;
        break;
      case Type::F64:
        store_func = MakeF64Store;
        break;
      default:
        assert(!"calling fill function with unsupported type");
    }
    b.Insert(store_func(MakeLocalGet(offset), MakeLocalGet(value), 1, 0));
    b.Insert(MakeLocalSet(offset, MakeBinary(Opcode::I32Add, MakeLocalGet(offset), MakeI32Const(type_size))));
    auto tee_local = MakeLocalTree(repeat, MakeBinary(Opcode::I32Sub, MakeLocalGet(repeat), MakeI32Const(1)));
    auto cond = MakeBinary(Opcode::I32Ne, tee_local, MakeI32Const(0));
    auto br_if = MakeBrIf(label, cond);
    b.Insert(br_if);
  });
  Merge(e, loop);
  return e;
}

#define BUILD_FILL_FUNC(t1, t2)                                                                             \
Var MemoryBuiltins::BuildFill##t1() {                                                                       \
  return module_manager_->MakeFunction(nullptr, {{Type::I32, Type::I32, Type::t1}, {}}, {},                 \
                                       [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {  \
    f.Insert(Fill<Type::t1>(&f, params[0], params[1], params[2]));                                          \
  });                                                                                                       \
}                                                                                                           \
Var MemoryBuiltins::Fill##t1() const {                                                                      \
  assert(options_->memory.fill_##t2);                                                                       \
  return fill_##t2##_;                                                                                      \
}
BUILD_FILL_FUNC(I32, i32);
BUILD_FILL_FUNC(I64, i64);
BUILD_FILL_FUNC(F32, f32);
BUILD_FILL_FUNC(F64, f64);

} // namespace wasmpp