import cudarray as ca
import deeppy as dp
import deeppy.expr as ex


class ScaleGradient(ex.base.UnaryElementWise):
    def __init__(self, scale):
        self.scale = scale

    def fprop(self):
        self.array = self.x.array

    def bprop(self):
        ca.multiply(self.grad_array, self.scale, self.x.grad_array)


class WeightedParameter(dp.Parameter):
    def __init__(self, parameter, weight, shared_weight):
        self.__dict__ = parameter.__dict__
        self.weight = weight
        self.shared_weight = shared_weight

    def grad(self):
        grad = self.grad_array
        grad *= self.weight
        for param in self.shares:
            p_grad_array = param.grad_array
            p_grad_array *= self.shared_weight
            grad += p_grad_array
        grad = self._add_penalty(grad)
        return grad
