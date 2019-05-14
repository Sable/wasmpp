#include <src/ir.h>
#include <src/wat-writer.h>
#include <src/stream.h>
#include <src/cast.h>

int main() {
  wabt::ExprList list;
  list.push_back(wabt::MakeUnique<wabt::NopExpr>());
  wabt::Module module;
  auto field = wabt::MakeUnique<wabt::FuncModuleField>();
  module.AppendField(std::move(field));
  wabt::WriteWatOptions watOptions;
  auto stream = wabt::FileStream(stdout);
  wabt::WriteWat(&stream, &module, watOptions);

  return 0;
}
