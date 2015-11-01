import numpy
import KTimage as KT
import math

# directory for observing the results
path_obs = "res/"


class MLP:

    """Creates a new instance of class MLP
    @size_in: number of input neurons
    @size_out: number of output neurons
    @size_hid: number of hidden layer neurons"""
    def __init__(self, size_in, size_out, size_hid):
        self.size_in = size_in
        self.size_out = size_out
        self.size_hid = size_hid

        # the activation function is a sigmoid function
        self.activation = MLP.activation_sigmoid
        self.activation_deriv = MLP.activation_sigmoid_deriv

        # create randomly initialized weight matrices for the hidden and the output layer
        self.weights_1 = numpy.random.uniform(-1.0, 1.0, (self.size_in, self.size_hid))
        self.weights_2 = numpy.random.uniform(-1.0, 1.0, (self.size_hid, self.size_out))

    """applies the MLP network to a set of input data and returns a list of the outputs of each perceptron layer"""
    def feedforward(self, data):
        results = [data]
        results.append(
            # apply hidden layer on input
            self.activation(numpy.dot(results[0], self.weights_1)))
        results.append(
            # apply output layer on input
            self.activation(numpy.dot(results[1], self.weights_2)))

        return results

    """calculates and applies the change in weight according to the error resulting from feedforward and the
    learning rate provided
    returns the absolute error
    @results: An array of layer wise results (as produced by feedforward)
    @target_output: A vector of output values to be used as training reference
    @learning_rate: Influences the performance (the higher the better) and accuracy (the lower the better)"""
    def backpropagate(self, results, target_output, learning_rate):

        # calculate output error
        error = target_output - results[2]

        delta2 = error * self.activation_deriv(results[2])
        delta1 = delta2.dot(self.weights_2.T) * self.activation_deriv(results[1])

        # adjust weights
        self.weights_1 += learning_rate * numpy.atleast_2d(results[0]).T.dot(numpy.atleast_2d(delta1))
        self.weights_2 += learning_rate * numpy.atleast_2d(results[1]).T.dot(numpy.atleast_2d(delta2))

        # returns the absolute error (distance of target output and actual output)
        return abs(error[0])

    """Trains the MLP with the given training data
    @t_input: Set of training input vectors
    @t_output: Set of training output vectors matching t_input
    @iterations: number of training epochs
    @learning_rate: Influences the performance (the higher the better) and accuracy (the lower the better)"""
    def train(self, t_input, t_output, iterations=10000, learning_rate=0.01):
        t_input = numpy.atleast_2d(t_input)
        for i in range(iterations):
            errors = []
            for j in numpy.nditer(t_input):
                errors.append(self.backpropagate(self.feedforward(t_input[j]), t_output[j], learning_rate))
            if not i % 100:
                print "Error: ", (sum(errors) / float(len(errors))), "\n"

    """Returns predicted output vector for a given input vector data_in"""
    def predict(self, data_in):
        return self.feedforward(data_in)[-1]

    @staticmethod
    def activation_sigmoid(x):
        return 1 / (1 + math.e**(-x))

    @staticmethod
    def activation_sigmoid_deriv(x):
        return MLP.activation_sigmoid(x) * (1 - MLP.activation_sigmoid(x))

    @staticmethod
    def activation_linear(x):
        return x

    @staticmethod
    def activation_linear_deriv(x):
        return 1

    @staticmethod
    def activation_tanh(x):
        return numpy.tanh(x)

    @staticmethod
    def activation_tanh_deriv(x):
        return 1.0 - numpy.tanh(x)**2


data_in_xor = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out_xor = numpy.array([[0], [1], [1], [0]])

data_in_and = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out_and = numpy.array([[0], [0], [0], [1]])

data_in_or = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out_or = numpy.array([[0], [1], [1], [1]])

data_in = data_in_xor
data_out = data_out_xor

mlp = MLP(2, 1, 3)

mlp.train(data_in, data_out)
