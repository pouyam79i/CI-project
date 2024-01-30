from rsdl.optim import Optimizer
from rsdl import Tensor
from rsdl.tensors import ensure_tensor
from rsdl.layers import Linear
import numpy as np

# TODO: implement Adam optimizer like SGD
class Adam(Optimizer):
    def __init__(self, layers: list[Linear], learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, time = 1):
        super().__init__(layers)
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time = time
        
    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        def f_adam(x: Tensor) -> Tensor:
            # initial momentum
            f_moment = Tensor(np.zeros_like(x.data))
            s_moment = Tensor(np.zeros_like(x.data))
            
            # update w, b momentum
            f_moment = (f_moment * self.beta1) + (x.grad * (1 - self.beta1))
            s_moment = (s_moment * self.beta2) + (x.grad * (1 - self.beta2))

            # bias correction
            f_ub = ensure_tensor(f_moment * (1/(1 - (self.beta1**self.time) + self.epsilon)))
            s_ub = ensure_tensor(s_moment * (1/(1 - (self.beta2**self.time) + self.epsilon)))
            
            # update w and b
            a = f_ub * self.learning_rate 
            b = (s_ub**0.5) + self.epsilon
            b.data = 1 / b.data
            update = a*b
    
            return update
            
        for l in self.layers:
            l.weight = l.weight - f_adam(l.weight)
            if l.need_bias:
                l.bias = l.bias - f_adam(l.bias)
                
        # update time step
        self.time = self.time + 1
