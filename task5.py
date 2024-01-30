# Task 5
import numpy as np

import matplotlib.pyplot as plt

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD, Adam
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([[-7], [+3], [-9]]))
y = X @ coef + 5


# TODO: define a linear layer using Linear() class  
l = Linear(3, 1, True)

# TODO: define an optimizer using SGD() class 
optimizer = Adam([l], learning_rate=0.01)

# TODO: print weight and bias of linear layer
print("initial W and bias:")
print(l.weight)
print(l.bias)

batch_size = 20
epLoss = []

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        # TODO: predicted
        # l.zero_grad()
        predicted = l.forward(inputs)

        actual = y[start:end]
        # actual.data = actual.data.reshape(batch_size, 1)
        # TODO: calcualte MSE loss
        loss = loss_functions.MeanSquaredError(predicted, actual)
        
        # TODO: backward
        # hint you need to just do loss.backward()
        loss.backward()

        # TODO: add loss to epoch_loss
        epoch_loss += loss.data

        # TODO: update w and b using optimizer.step()
        optimizer.step()        

    epLoss.append(epoch_loss)
    
# TODO: print weight and bias of linear layer
print("OUT W and bias:")
print(l.weight)
print(l.bias)

plt.plot(epLoss, label='Error')
plt.xlabel('epoch')
plt.ylabel('error')
plt.show();
