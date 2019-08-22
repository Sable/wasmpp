# Features

## Activation Functions
Activation function | Layer usage       | JavaScript| C++
--------------------|-------------------|-----------|----
Sigmoid             | Hidden and Output | sigmoid   | Sigmoid()
Softmax             | Output            | softmax   | Softmax()
ReLU                | Hidden            | relu      | ReLU()
LeakyReLU           | Hidden            | leakyrelu | LeakyReLU()
ELU                 | Hidden            | elu       | ELU()
Tanh                | Hidden            | tanh      | Tanh()

## Loss Functions
Loss function         | Output layer activation function | JavaScript            | C++
----------------------|----------------------------------|-----------------------|----
Mean Squared Error    | Sigmoid                          | mean-squared-error    | MeanSquaredError()
Sigmoid Cross Entropy | Sigmoid                          | sigmoid-cross-entropy | SigmoidCrossEntropy()
Softmax Cross Entropy | Softmax                          | softmax-cross-entropy | SoftmaxCrossEntropy()

## Weight Initialization
Weight initializer | Layer Position    | JavaScript     | C++
-------------------|-------------------|----------------|----
Xavier Uniform     | Hidden            | xavier_uniform | XavierUniform()
Xavier Normal      | Hidden            | xavier_normal  | XavierNormal()
LeCun Uniform      | Hidden and Output | lecun_uniform  | LeCunUniform()
LeCun Normal       | Hidden and Output | lecun_normal   | LeCunNormal()
Gaussian           | Hidden and Output | gaussian       | Gaussian()
Uniform            | Hidden and Output | uniform        | Uniform()
Constant           | Hidden and Output | constant       | Constant()

## Weights Optimizer
Weights Optimizer |
----------------- | 
 Stochastic Gradient Descent |

## Regularization
Regularization | Scope
---------------|------
Dropout        | Input layer and Hidden layers
L1             | All weights and bias values
L2             | All weights and bias values
