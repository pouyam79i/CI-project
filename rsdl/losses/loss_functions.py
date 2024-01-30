from rsdl import Tensor
import numpy as np

#TODO: split into data, grad, dependency ... then build tensor at the end!
def MeanSquaredError(preds: Tensor, actual: Tensor) -> Tensor:
    # TODO : implement mean squared error
    return ((preds - actual)**2.0).sum() * (1 / actual.data.shape[0])


def CategoricalCrossEntropy(preds: Tensor, actual: Tensor) -> Tensor:
    # TODO : imlement categorical cross entropy 
    return ((actual * (preds.log())) + (1 - actual) * ((1 - preds).log())).sum()
