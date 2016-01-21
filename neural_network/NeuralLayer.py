import numpy as np
from pycuda import gpuarray
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
from lstm_network.utils import debug


culinalg.init()


class NeuralLayer(object):
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        self.weights = gpuarray.to_gpu(np.random.uniform(-1.0, 1.0, (out_size, in_size)))
        self.biases = gpuarray.zeros((out_size, 1))
        self.activation = activation_fn
        self.activation_deriv = activation_fn_deriv
        self.size = out_size

    def feed(self, data):
        #print("weights shape: " + str(numpy.shape(self.weights)) + " - data shape: " + str(numpy.shape(data)) + " - bias shape: " + str(numpy.shape(self.biases)))
        return self.activation(culinalg.dot(self.weights, data) + self.biases.T)

    def reverse_feed(self, data):
        return self.activation(culinalg.dot(data - self.biases, self.weights))

    def learn(self, result, delta, learning_rate):
        delta_weights = learning_rate * culinalg.outer(delta, result)
        self.weights += delta_weights

    def get_delta(self, result, last_delta, last_weights):
        debug("delta shape: " + str(last_delta.shape) + " - result shape: " + str(numpy.shape(result)) + " - weights shape: " + str(numpy.shape(self.weights)))
        last_weights = last_weights
        return culinalg.dot(last_delta, last_weights) * self.activation_deriv(result)


    @classmethod
    def activation_linear(cls, x):
        return x

    @classmethod
    def activation_linear_deriv(cls, x):
        return 1

    @classmethod
    def activation_sigmoid(cls, x):
        return 1 / (1 + numpy.exp(-x))

    @classmethod
    def activation_sigmoid_deriv(cls, x):
        return (1. - x) * x

    @classmethod
    def activation_tanh(cls, x):
        return numpy.tanh(x)

    @classmethod
    def activation_tanh_deriv(cls, x):
        return 1.0 - x ** 2


class BiasedNeuralLayer(NeuralLayer):
    def learn(self, result, delta, learning_rate):
        super(BiasedNeuralLayer, self).learn(result, delta, learning_rate)
        self.biases += learning_rate * delta
