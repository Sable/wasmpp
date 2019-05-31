import numpy as np
from tabulate import tabulate

class Model:
    W = []
    Z = []
    A = []
    B = []
    dW = []
    dZ = []
    dA = []
    dB = []
    T = 0
    alpha = 0.01

    def sigmoid(self, x, prime):
        if prime:
            s = self.sigmoid(x, False)
            return s * (1 - s)
        return 1 / (1 + np.exp(-x))

    activation = sigmoid

    def mse(self, t, p, prime):
        if prime:
            return p - t
        return 0.5 * (t - p)**2

    loss = mse

    def build(self, model, alpha):
        self.model = model
        self.alpha = alpha
        prev = 1
        for i in range(0, len(model)):
            self.A.append(np.zeros(shape=(model[i],1)))
            self.dA.append(np.zeros(shape=(model[i],1)))
            self.Z.append(np.zeros(shape=(model[i],1)))
            self.dZ.append(np.zeros(shape=(model[i],1)))
            self.W.append(np.full((model[i],prev), 0.1))
            self.dW.append(np.zeros(shape=(model[i],prev)))
            self.B.append(np.full((model[i],1), 0.1))
            self.dB.append(np.zeros(shape=(model[i],1)))
            prev = model[i]

    def copy_input(self, data, label):
        self.A[0] = np.array(data).reshape(len(data), 1)
        self.T = np.array(label).reshape(len(label), 1)

    def forward(self):
        for l in range(1, len(self.model)):
            # Z[l] = W[l] . A[l-1] + B[l]
            self.Z[l] = self.W[l].dot(self.A[l-1]) + self.B[l]
            # A[l] = g(Z[l])
            self.A[l] = self.activation(self.Z[l], False)
        # dA[L] = Loss(T, A[L])
        self.dA[len(self.model)-1] = self.loss(self.T, self.A[len(self.model)-1], True)

    def backward(self):
        for l in range(len(self.model)-1, 0, -1):
            # dZ[l] = dA[l] * g'(Z[l])
            self.dZ[l] = self.dA[l] * self.activation(self.Z[l], True)
            # dW[l] = (1/m) dZ[l] . A[l-1]^T
            self.dW[l] = (1/1) * (self.dZ[l].dot(self.A[l-1].T))
            # dB[l] = (1/m) dZ[l]
            self.dB[l] = (1/1) * self.dZ[l]
            # dA[l-1] = W[l]^T . dZ[l]
            self.dA[l-1] = self.W[l].T.dot(self.dZ[l])
            # W[l] = W[l] - alpha * dW[l]
            self.W[l] = self.W[l] - (self.alpha * self.dW[l])
            # B[l] = B[l] - alpha * dB[l]
            self.B[l] = self.B[l] - (self.alpha * self.dB[l])

train = [[0,0],
        [0,1],
        [1,0],
        [1,1]]

train_label = [[1,0],
        [0,1],
        [0,1],
        [0,1]]

m = Model()
model = [2, 2, 2]
alpha = 0.01
epoch = 10000

m.build(model, alpha)
for e in range(0, epoch):
    for i in range(0, len(train)):
        m.copy_input(train[i], train_label[i])
        m.forward()
        m.backward()

for i in range(0, len(train)):
    m.copy_input(train[i], train_label[i])
    m.forward()
    print tabulate(m.A[len(m.A)-1])
