#include <src/apps/nn-builder/src/nn-math.h>

namespace nn {

using namespace wabt;

Var MathExtension::Exp() {
  std::string func_name = "exp";
  if(func_map.find(func_name) != func_map.end()) return func_map[func_name];
  Var func_var = mb_->CreateFunction(func_name.c_str(), {{Type::F64}, {Type::F64}}, {},
                            [&](ExprList* e, std::vector<Var> params, std::vector<Var> locals) {

  });
  func_map[func_name] = func_var;
}

Var MathExtension::Sigmoid() {
  std::string func_name = "sigmoid";
  if(func_map.find(func_name) != func_map.end()) return func_map[func_name];
  Var func_var = mb_->CreateFunction(func_name.c_str(), {{Type::F64}, {Type::F64}}, {},
                            [&](ExprList* e, std::vector<Var> params, std::vector<Var> locals) {

  });
  func_map[func_name] = func_var;
}

} // namespace nn