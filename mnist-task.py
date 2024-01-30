# TODO: in this task you have to
# 1. load mnist dataset for our framework
# 2. define your model
# 3. start training and have fun!

# TODO: remove what is not needed
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader

# TODO: import new ones here *****************
from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.activations import Relu, Softmax
from rsdl.optim import SGD
from rsdl.losses import CategoricalCrossEntropy
import numpy as np

train_set = datasets.MNIST(
    '../data', train=True, download=True, transform=Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
)

test_set = datasets.MNIST(
    '../data', train=False, transform=Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
)

# TODO: define dataloader for train and test
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

# Here in this implementation we used our defined lib :) 
class Model():
    def __init__(self):
        self.fc_in = Linear(784, 112)
        self.fc_out = Linear(112, 10)
        
    def forward(self, x):
        x = torch.flatten(x, 1).numpy() # we need to flatten data so just a little using of torch here
        x = self.fc_in.forward(Tensor(x))
        x = Relu(x)
        x = self.fc_out.forward(x)
        res = Softmax(x)
        return res
    
    def getLayers(self):
        return [self.fc_in, self.fc_out]

# TODO: define your model dont forget about device :)
model = Model()

# TODO: define optimizer
optimizer = SGD(model.getLayers(), learning_rate=0.01)

# TODO: define loss
criterion = CategoricalCrossEntropy

# TODO: make it work :) 
def train_one_epoch(model:Model, data, optimizer, criterion):
  for images, labels in data:
    labels = labels.numpy()
    optimizer.zero_grad()
    yp = model.forward(images)
    loss = criterion(Tensor(np.argmax(yp.data, axis=1), requires_grad=yp.grad, depends_on=yp.depends_on), labels)
    loss.backward()
    optimizer.step()
  return loss

# training process
accs = []
best_acc = 0
for e in range(20):
  accs.append(train_one_epoch(model, train_loader, optimizer, criterion))

print(accs)
