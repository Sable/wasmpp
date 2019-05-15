#ifndef NN_BUILDER_MATH_H
#define NN_BUILDER_MATH_H

namespace nn {
  class MathModule {
  private:
    wasmpp::ModuleBuilder mb_;

    // Functions vars
    wabt::Var f_sigmoid_;
    wabt::Var f_exp_;

    // Sigmoid function
    void CreateSigmoid();

    // Exp function
    void CreateExo();
  public:
    // Initiliaze module
    void Init();
  };

} // namespace nn

#endif