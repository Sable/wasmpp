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
    Cost = 0
    Conf_matrix = 0

    def sigmoid(self, x, prime):
        if prime:
            s = self.sigmoid(x, False)
            return s * (1 - s)
        return 1 / (1 + np.exp(-x))


    def mse(self, t, p, prime):
        if prime:
            return p - t
        return 0.5 * (t - p)**2

    def cost_function(self):
        cost = self.loss(self.T, self.A[-1], False)
        return cost.mean()

    def confusion_matrix(self):
        pred = np.argmax(self.A[-1], axis=0)[0]
        real = np.argmax(self.T, axis=0)[0]
        self.Conf_matrix[real][pred] += 1

    def build(self, model, acts, loss, alpha):
        self.model = model
        self.alpha = alpha
        self.acts = acts
        self.loss = loss
        self.Conf_matrix = np.zeros(shape=(model[-1], model[-1]))
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
            self.A[l] = self.acts[l](self.Z[l], False)
        # dA[L] = Loss(T, A[L])
        self.dA[-1] = self.loss(self.T, self.A[-1], True)

    def backward(self):
        for l in range(len(self.model)-1, 0, -1):
            # dZ[l] = dA[l] * g'(Z[l])
            self.dZ[l] = self.dA[l] * self.acts[l](self.Z[l], True)
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
