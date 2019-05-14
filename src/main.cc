#include <src/ir.h>

int main() {
  wabt::ExprList list;
  list.push_back(wabt::MakeUnique<wabt::NopExpr>());
  return 0;
}
