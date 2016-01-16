import numpy
from lstm_network.utils import debug


class NeuralLayer(object):
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        self.weights = numpy.random.uniform(-1.0, 1.0, (out_size, in_size))
        self.biases = numpy.zeros(out_size)
        self.activation = activation_fn
        self.activation_deriv = activation_fn_deriv
        self.size = out_size

    def feed(self, data):
        #print("weights shape: " + str(numpy.shape(self.weights)) + " - data shape: " + str(numpy.shape(data)) + " - bias shape: " + str(numpy.shape(self.biases)))
        return self.activation(numpy.dot(self.weights, data) + numpy.atleast_2d(self.biases).T)

    def reverse_feed(self, data):
        return self.activation(numpy.dot(data - numpy.atleast_2d(self.biases), self.weights))

    def learn(self, result, delta, learning_rate):
        delta_weights = learning_rate * numpy.outer(delta, result)
        self.weights += delta_weights

    def get_delta(self, result, last_delta, last_weights):
        debug("delta shape: " + str(numpy.shape(last_delta)) + " - result shape: " + str(numpy.shape(result)) + " - weights shape: " + str(numpy.shape(self.weights)))
        last_weights = numpy.atleast_2d(last_weights)
        return numpy.dot(last_delta, last_weights) * self.activation_deriv(result)


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
