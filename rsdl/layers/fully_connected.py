from rsdl import Tensor
from rsdl.layers import initializer

class Linear:

    def __init__(self, in_channels, out_channels, need_bias=True, mode='xavier') -> None:
        # set input and output shape of layer
        self.shape = (in_channels, out_channels)
        self.need_bias = need_bias
        # TODO initialize weight by initializer function (mode)
        self.weight = Tensor(
            data=initializer(shape=self.shape, mode=mode),
            requires_grad=True
        )
        # TODO initialize weight by initializer function (zero mode)
        if self.need_bias:
            self.shape = (1, out_channels)
            self.bias = Tensor(
                data=initializer(shape=self.shape, mode="zero"),
                requires_grad=True
            )

    def forward(self, inp: Tensor) -> Tensor:
        # TODO:implement forward propagation
        out = inp @ self.weight
        if self.need_bias:
            out = out + self.bias
        return out
    
    def parameters(self):
        if self.need_bias:
            return [self. weight, self.bias]
        return [self.weight]
    
    def zero_grad(self):
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()
            
    def __call__(self, inp: Tensor) -> Tensor:
        return self.forward(inp)
