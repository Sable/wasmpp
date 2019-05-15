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
  auto loop = mb.CreateLoop([&](wabt::ExprList* e, wabt::Var label) {
    auto rhs = mb.CreateI32Const(1);
    auto a = mb.GenerateBranchIfCompInc(label, wabt::Opcode::I32Ne, wabt::Var("$k"), 1, &rhs);
    wasm::ModuleBuilder::Merge(e, &a);
  });
  mb.CreateFunction("sigmoid", &loop);

  // Write module to output stream
  wabt::WriteWatOptions watOptions;
  watOptions.fold_exprs = true;
  watOptions.inline_import = false;
  watOptions.inline_export = true;
  auto stream = wabt::FileStream(stdout);
  wabt::WriteWat(&stream, &module, watOptions);

  return 0;
}
