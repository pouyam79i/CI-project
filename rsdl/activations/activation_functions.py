from rsdl import Tensor, Dependency
import numpy as np


# Implementing some aux. functions here:
def s(x:np.ndarray) -> np.ndarray:
    y = 1.0 + np.exp(np.negative(x))
    y = 1.0 / y
    return y

def th(x: np.ndarray) -> np.ndarray:
    ep = np.exp(x)
    ep_neg = np.exp(np.negative(x))
    y = (ep - ep_neg) / (ep + ep_neg)
    return y
# End of aux. functions

def Sigmoid(t: Tensor) -> Tensor:
    # TODO: implement sigmoid function
    data = s(t.data)
    
    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * ((1 - s(t.data)) * s(t.data))
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(
        data=data,
        requires_grad=req_grad,
        depends_on=depends_on
    )

def Tanh(t: Tensor) -> Tensor:
    # TODO: implement tanh function
    # hint: you can do it using function you've implemented (not directly define grad func)
    data = th(t.data)
    
    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * (1 - np.power(th(t.data), 2))
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
        
    return Tensor(
        data=data,
        requires_grad=req_grad,
        depends_on=depends_on
    )

def Softmax(t: Tensor) -> Tensor:
    # TODO: implement softmax function
    # hint: you can do it using function you've implemented (not directly define grad func)
    # hint: you can't use sum because it has not axis argument so there are 2 ways:
    #        1. implement sum by axis
    #        2. using matrix mul to do it :) (recommended)
    # hint: a/b = a*(b^-1)
    data = t.data
    sum = t.exp().sum().data
    data = data / sum

    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            p1 = t.data
            p2 = np.ones_like(p1) * sum
            p2 = p2 - p1
            p1 = p1 * p2 
            dy = p1 / pow(sum, 2)
            return grad * dy
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(
        data=data,
        requires_grad=req_grad,
        depends_on=depends_on
    )

def Relu(t: Tensor) -> Tensor:
    # TODO: implement relu function
    # use np.maximum
    data = t.data
    data = np.maximum(data, 0)
    
    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(data > 0, 1, 0)
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
        
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def LeakyRelu(t: Tensor,leak=0.05) -> Tensor:
    """
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn 
    hint: use np.where like Relu method but for LeakyRelu
    """
    # TODO: implement leaky_relu function
    data = t.data
    data = np.maximum(data, leak * data)
    
    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(data > 0, 1, leak)
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
        
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
