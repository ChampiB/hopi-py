import numpy as np
import math
import scipy


class Gamma:

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def params(self):
        params = np.zeros(2)
        params[0] = - self.scale
        params[1] = self.shape - 1
        return params

    def set_params(self, params):
        self.scale = - params[0]
        self.shape = params[1] + 1

    def probability(self, x):
        return math.pow(self.scale, self.shape) / math.gamma(self.shape) * math.pow(x, self.shape - 1) * \
               math.exp(- self.scale * x)

    def hidden_expectations(self):
        mpmath.psi(m, z)
        params = self.params()
        msg = np.zeros(2)
        msg[0] = - (params[1] + 1) / params[0]
        msg[1] = - math.log(- params[0]) - 1 / params[1] - scipy.special.digamma(params[1])
        return msg
