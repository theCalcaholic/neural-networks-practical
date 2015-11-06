import numpy
import math


class PerceptronLayer:
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        self.weights = numpy.random.uniform(-1.0, 1.0, (in_size, out_size))
        self.activation = activation_fn
        self.activation_deriv = activation_fn_deriv
        self.size = in_size

    def feed(self, data):
        return self.activation(numpy.dot(data, self.weights))

    def learn(self, result, delta, learning_rate):
        #self.weights += learning_rate * numpy.outer(result, delta)
        self.weights += learning_rate * numpy.atleast_2d(result).T.dot(numpy.atleast_2d(delta))


    @classmethod
    def activation_linear(cls, x):
        return x

    @classmethod
    def activation_linear_deriv(cls, x):
        return 1

    @classmethod
    def activation_sigmoid(cls, x):
        return 1 / (1 + math.e**(-x))

    @classmethod
    def activation_sigmoid_deriv(cls, x):
        return cls.activation_sigmoid(x) * (1 - cls.activation_sigmoid(x))

    @classmethod
    def activation_tanh(cls, x):
        return numpy.tanh(x)

    @classmethod
    def activation_tanh_deriv(cls, x):
        return 1.0 - numpy.tanh(x)**2