import numpy
from OutputLayer import OutputLayer
from HiddenLayer import HiddenLayer
from PerceptronLayer import PerceptronLayer
from SimpleMLP import MLP as SimpleMLP


# directory for observing the results
path_obs = "res/"


class MLP:

    """Creates a new instance of class MLP
    @size_in: number of input neurons
    @size_out: number of output neurons
    @size_hid: number of hidden layer neurons"""
    def __init__(self,
                 in_size=0,
                 out_size=0,
                 hidden_sizes=[], hidden_fns=[], hidden_fns_deriv=[],
                 learning_rate=0.01):
        self.populated = False
        self.trained = False
        self.learning_rate = learning_rate
        """self.size_in = size_in
        self.size_out = size_out
        self.size_hid = size_hid"""
        self.layers = []

        if in_size and out_size:
            self.populate(in_size, out_size, hidden_sizes, hidden_fns, hidden_fns_deriv)

        # the activation function is a sigmoid function
        """self.activation = MLP.activation_sigmoid
        self.activation_deriv = MLP.activation_sigmoid_deriv"""

        # create randomly initialized weight matrices for the hidden and the output layer
        """self.weights_1 = numpy.random.uniform(-1.0, 1.0, (self.size_in, self.size_hid))
        self.weights_2 = numpy.random.uniform(-1.0, 1.0, (self.size_hid, self.size_out))"""

    def populate(self, in_size,
                 out_size,
                 hidden_sizes=[], hidden_fns=[], hidden_fns_deriv=[]):
        print("populate!")
        self.layers = []
        self.layers.append(HiddenLayer(in_size, hidden_sizes[0], hidden_fns[0], hidden_fns_deriv[0]))
        if len(hidden_sizes) != 0:
            for cur_size, fn, fn_deriv in zip(hidden_sizes[1:], hidden_fns[1:], hidden_fns_deriv[1:]):
                self.layers.append(HiddenLayer(
                    in_size=self.layers[-1].size,
                    out_size=cur_size,
                    activation_fn=fn,
                    activation_fn_deriv=fn_deriv))
        self.layers.append(OutputLayer(
            in_size=self.layers[-1].size,
            out_size=out_size))
        self.populated = True

    """applies the MLP network to a set of input data and returns a list of the outputs of each perceptron layer"""
    def feedforward(self, data):
        results = [data]

        for layer in self.layers:
            results.append(
                layer.feed(results[-1])
            )

        """results.append(
            # apply hidden layer on input
            self.activation(numpy.dot(results[0
            ], self.weights_1)))
        results.append(
            # apply output layer on input
            self.activation(numpy.dot(results[1], self.weights_2)))"""

        return results

    """calculates and applies the change in weight according to the error resulting from feedforward and the
    learning rate provided
    returns the absolute error
    @results: An array of layer wise results (as produced by feedforward)
    @target_output: A vector of output values to be used as training reference
    @learning_rate: Influences the performance (the higher the better) and accuracy (the lower the better)"""
    def backpropagate(self, results, target_output):

        # calculate output error
        deltas = [target_output - results[-1]]

        for result, layer, last_weights in reversed(zip(
            results[1:],
            self.layers[:-1],
            [l.weights for l in self.layers[1:]]
        )):

            deltas.append(
                layer.get_delta(
                    result,
                    deltas[-1],
                    last_weights
                )
            )


        """delta2 = error * self.activation_deriv(results[2])
        delta1 = delta2.dot(self.weights_2.T) * self.activation_deriv(results[1])"""

        for result, delta, layer in zip(results[:-1], reversed(deltas), self.layers):
            layer.learn(result, delta, self.learning_rate)

        #self.bias_1 = min(0, -(learning_rate * delta1) + self.bias_1)
        #self.bias_2 = min(0, -(learning_rate * delta2) + self.bias_2)

        # returns the absolute error (distance of target output and actual output)
        return deltas[0][0]**2

    """Trains the MLP with the given training data
    @t_input: Set of training input vectors
    @t_output: Set of training output vectors matching t_input
    @iterations: number of training epochs
    @learning_rate: Influences the performance (the higher the better) and accuracy (the lower the better)"""
    def train(self, t_input, t_output, iterations=10000):
        t_input = t_input
        for i in range(iterations):
            errors = []
            for j in numpy.nditer(t_input):
                errors.append(self.backpropagate(self.feedforward(t_input[j]), t_output[j]))
            if not i % 10:
                print "Error: ", (sum(errors) / float(len(errors))), "\n"

    """Returns predicted output vector for a given input vector data_in"""
    def predict(self, data_in):
        return self.feedforward(data_in)[-1]

    def print_layers(self):
        for i in range(len(self.layers)):
            print("Layer " + str(i) + " (" + self.layers[i].layer_type + "):")
            print("  shape: " + str(numpy.shape(self.layers[i].weights)))
            print("  activation_fn: " + str(self.layers[i].activation))

    def set_learning_rate(self, val):
        self.learning_rate = val
