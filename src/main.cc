#include <src/ir.h>
#include <src/wat-writer.h>
#include <src/stream.h>
#include <src/cast.h>
#include <src/wasm/wasm-builder.h>

int main() {

  // Build module
  wasm::ModuleBuilder mb;
  auto& module = mb.GetModule();

  // Create function
  mb.CreateFunction("sigmoid", {{wabt::Type::I32, wabt::Type::I32}, {wabt::Type::I32}}, {wabt::Type::I32},
                    [&](wabt::ExprList* e, std::vector<wabt::Var> params, std::vector<wabt::Var> locals) {
    auto loop = mb.CreateLoop([&](wabt::ExprList* e, wabt::Var label) {
      auto inc = mb.CreateI32Const(1);
      auto rhs = mb.CreateI32Const(1);
      auto a = mb.GenerateBranchIfCompInc(label, wabt::Type::I32, wabt::Opcode::I32Ne, params[1], &inc, &rhs);
      wasm::ModuleBuilder::Merge(e, &a);
    });
    wasm::ModuleBuilder::Merge(e, &loop);
  });


  // Write module to output stream
  wabt::WriteWatOptions watOptions;
  watOptions.fold_exprs = true;
  watOptions.inline_import = false;
  watOptions.inline_export = true;
  auto stream = wabt::FileStream(stdout);
  wabt::WriteWat(&stream, &module, watOptions);

  return 0;
}
