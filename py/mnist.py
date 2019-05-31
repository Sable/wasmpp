from tabulate import tabulate
from nn import Model
import sys
import csv

train = []
train_label = []
test = []
test_label = []
train_limit = 1000
test_limit = 10
with open(sys.argv[1]) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    r = 0
    for row in readCSV:
        label = [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0]
        if r > 0:
            label[int(row[0])] = 1
            if len(train) < train_limit:
                train_label.append(label)
                train.append(row[1:])
                train[-1] = map(lambda x: float(x), train[-1])
            elif len(test) < test_limit:
                test_label.append(label)
                test.append(row[1:])
                test[-1] = map(lambda x: float(x), test[-1])
        r += 1

m = Model()
epoch = 100
alpha = 0.1
model = [784, 100, 10]
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
for i in range(0, len(test)):
    m.copy_input(test[i], test_label[i])
    m.forward()
    cost += m.cost_function()
    m.confusion_matrix()
print("Testing error : {}"
        .format(cost / len(train)))
print tabulate(m.Conf_matrix)
