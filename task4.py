# Task 4
import numpy as np

import matplotlib.pyplot as plt

from rsdl import Tensor
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([[-7], [+3], [-9]]))
y = X @ coef + 6

# TODO: define w and b (y = x w + b) with random initialization ( you can use np.random.randn )
w = Tensor(np.random.randn(3, 1), requires_grad=True)
b = Tensor(np.array([0]), requires_grad=True)
print("initial W and bias:")
print(w)
print(b)

learning_rate = 0.01
batch_size = 20
epLoss = []

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]
        # TODO: predicted
        w.zero_grad()
        b.zero_grad()
        predicted = (inputs @ w) + b

        actual = y[start:end]
        # TODO: calcualte MSE loss
        loss = loss_functions.MeanSquaredError(predicted, actual)
        
        # TODO: backward
        # hint you need to just do loss.backward()
        loss.backward()

        epoch_loss += loss.data
        
        # TODO: update w and b (Don't use 'w -= ' and use ' w = w - ...') (you don't need to use optim.SGD in this task)
        w = w - (learning_rate * w.grad)
        b = b - (learning_rate * b.grad)
        # print("R: {}, E: {}".format(epoch, start))

    # print("Loss:", epoch_loss)
    epLoss.append(epoch_loss)
    
# print("Loss: ", epLoss)    
print("model last W and bias:")
print(w)
print(b)

plt.plot(epLoss, label='Error')
plt.xlabel('epoch')
plt.ylabel('error')
plt.show();
