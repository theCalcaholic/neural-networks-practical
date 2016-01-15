import numpy
from LayerCache import LayerCache


class NeuralLayer(object):
    def __init__(self, in_size, out_size, predecessor=None, successor=None, activation_fn=None, activation_fn_deriv=None):
        self.weights = numpy.random.uniform(-1.0, 1.0, (out_size, in_size))
        self.biases = numpy.zeros(out_size)
        self.activation = activation_fn or NeuralLayer.activation_linear
        self.activation_deriv = activation_fn_deriv or NeuralLayer.activation_linear_deriv
        self.size = out_size
        self.cache = LayerCache()
        self.predecessor = predecessor
        self.successor = successor
        self.cache = LayerCache

    def feed(self, data):
        self.cache.input = data
        self.cache.output = self.activation(numpy.dot(self.weights, data) + self.biases)
        if self.successor:
            self.successor.feed(self.cache.output)
        else:
            return self.cache.output

    def learn(self, result, delta, learning_rate):
        delta_weights = learning_rate * numpy.outer(delta, result)
        self.weights += delta_weights

    def get_delta(self, result, last_delta, last_weights):
        last_weights = numpy.atleast_2d(last_weights)
        self.cache.delta = numpy.dot(last_delta, last_weights) * self.activation_deriv(result)
        return self.cache.delta

    def add_predecessor(self, predecessor):
        self.predecessor = predecessor

    def add_successor(self, successor):
        self.successor = successor

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
        return x * (1 - x)

    @classmethod
    def activation_tanh(cls, x):
        return numpy.tanh(x)

    @classmethod
    def activation_tanh_deriv(cls, x):
        return 1.0 - numpy.tanh(x)**2


class BiasedNeuralLayer(NeuralLayer):
    def learn(self, result, delta, learning_rate):
        super(BiasedNeuralLayer, self).learn(result, delta, learning_rate)
        self.biases = min(0, -(learning_rate * delta) + self.biases)
