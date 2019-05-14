#include <src/ir.h>
#include <src/wat-writer.h>
#include <src/stream.h>
#include <src/cast.h>
#include <src/wasm/wasm-builder.h>

int main() {

  // Build module
  wasm::ModuleBuilder moduleBuilder;
  auto& module = moduleBuilder.GetModule();

  // Create function
  auto f = moduleBuilder.CreateFunction();
  f->exprs.push_back(wabt::MakeUnique<wabt::NopExpr>());

  // Write module to output stream
  wabt::WriteWatOptions watOptions;
  auto stream = wabt::FileStream(stdout);
  wabt::WriteWat(&stream, &module, watOptions);



  return 0;
}
