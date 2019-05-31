from tabulate import tabulate
from nn import Model
train = [[0,0],
        [0,1],
        [1,0],
        [1,1]]

train_label = [[1,0],
        [0,1],
        [0,1],
        [0,1]]

m = Model()
alpha = 0.01
epoch = 10000
model = [2, 2, 2]
acts = [m.sigmoid, m.sigmoid, m.sigmoid]

m.build(model, acts, m.mse, alpha)
for e in range(0, epoch):
    cost = 0
    for i in range(0, len(train)):
        m.copy_input(train[i], train_label[i])
        m.forward()
        cost += m.cost_function()
        m.backward()
    print("Training error at epoch {} : {}"
            .format(e+1, cost / len(train)))

cost = 0
for i in range(0, len(train)):
    m.copy_input(train[i], train_label[i])
    m.forward()
    cost += m.cost_function()
    m.confusion_matrix()
print("Testing error : {}"
        .format(cost / len(train)))
print tabulate(m.Conf_matrix)
