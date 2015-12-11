import numpy


class NeuralLayer(object):
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        self.weights = numpy.random.uniform(-1.0, 1.0, (out_size, in_size))
        self.biases = numpy.zeros(out_size)
        self.activation = activation_fn
        self.activation_deriv = activation_fn_deriv
        self.size = out_size

    def feed(self, data):
        return self.activation(numpy.dot(self.weights, data) + self.biases)

    def learn(self, result, delta, learning_rate):
        delta_weights = learning_rate * numpy.outer(delta, result)
        self.weights += delta_weights

    def get_delta(self, result, last_delta, last_weights):
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
        self.biases += learning_rate * delta
