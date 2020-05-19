import numpy as np
import math


class Gaussian:

    def __init__(self, mean, precision):
        self.mean = mean
        self.precision = precision

    def params(self):
        params = np.zeros(2)
        params[0] = self.mean * self.precision
        params[1] = - self.precision / 2
        return params

    def set_params(self, params):
        self.precision = -2 * params[1]
        self.mean = params[0] / self.precision

    def probability(self, x):
        return math.sqrt(self.precision / (2 * math.pi)) * math.exp(- self.precision * math.pow(x - self.mean, 2) / 2)

    @staticmethod
    def observed_expectations(x):
        msg = np.zeros(2)
        msg[0] = x
        msg[1] = math.pow(x, 2)
        return msg

    def hidden_expectations(self):
        params = self.params()
        msg = np.zeros(2)
        msg[0] = - params[0] / (2 * params[1])
        msg[1] = 1 / (2 * params[1]) + math.pow(params[0], 2) / (4 * math.pow(params[1], 2))
        return msg

    @staticmethod
    def mean_message(data_expectation, precision_expectation):
        msg = np.zeros(2)
        msg[0] = precision_expectation[0] * data_expectation[0]
        msg[1] = - precision_expectation[0] / 2
        return msg

    @staticmethod
    def precision_message(data_expectation, mean_expectation):
        msg = np.zeros(2)
        msg[0] = mean_expectation[0] * data_expectation[0] + data_expectation[2] / 2 - mean_expectation[1] / 2
        msg[1] = 1 / 2
        return msg
