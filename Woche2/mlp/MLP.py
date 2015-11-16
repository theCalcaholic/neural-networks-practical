import numpy
from OutputLayer import OutputLayer
from PerceptronLayer import PerceptronLayer as Layer


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
                 hidden_layers=None,
                 learning_rate=0.01):
        self.populated = False
        self.trained = False
        self.learning_rate = learning_rate
        self.layers = []
        self.hidden_layer_specs = hidden_layers

        if in_size and out_size:
            self.populate(in_size, out_size, hidden_layers)

    def populate(self, in_size,
                 out_size,
                 hidden_layers=None):
        hidden_layers = hidden_layers or self.hidden_layer_specs or []
        self.layers = []
        self.layers.append(Layer(
            in_size=in_size,
            out_size=hidden_layers[0]["size"],
            activation_fn=hidden_layers[0]["fn"] or Layer.activation_linear,
            activation_fn_deriv=hidden_layers[0]["fn_deriv"] or Layer.activation_linear_deriv))

        if len(hidden_layers) != 0:
            for cur_layer in hidden_layers[1:]:
                self.layers.append(Layer(
                    in_size=self.layers[-1].size,
                    out_size=cur_layer["size"],
                    activation_fn=cur_layer["fn"] or Layer.activation_sigmoid,
                    activation_fn_deriv=cur_layer["fn_deriv"] or Layer.activation_sigmoid_deriv))

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

        error = deltas[0][0]**2

        for result, delta, layer in zip(results[:-1], reversed(deltas), self.layers):
            layer.learn(result, delta, self.learning_rate)

        #self.bias_1 = min(0, -(learning_rate * delta1) + self.bias_1)
        #self.bias_2 = min(0, -(learning_rate * delta2) + self.bias_2)

        # returns the absolute error (distance of target output and actual output)
        return error

    """Trains the MLP with the given training data
    @t_input: Set of training input vectors
    @t_output: Set of training output vectors matching t_input
    @iterations: number of training epochs
    @learning_rate: Influences the performance (the higher the better) and accuracy (the lower the better)"""
    def train(self, t_input, t_output, iterations=100000):
        t_input = t_input
        for i in range(iterations):
            errors = []
            for j in range(0, numpy.shape(t_input)[0]):
                errors.append(self.backpropagate(self.feedforward(t_input[j]), t_output[j]))
            if not i % 100:
                print "Error: ", (sum(errors) / float(len(errors))), "\n"

    """Returns predicted output vector for a given input vector data_in"""
    def predict(self, data_in):
        return self.feedforward(data_in)[-1][0]

    def print_layers(self):
        for i in range(len(self.layers)):
            print("Layer " + str(i) + ":")
            print("  shape: " + str(numpy.shape(self.layers[i].weights)))
            print("  activation_fn: " + str(self.layers[i].activation))

    def add_layer(self, size, fn, fn_deriv):
        self.hidden_layers.append({
            "size": size,
            "fn": fn,
            "fn_deriv": fn_deriv
        })

    def set_learning_rate(self, val):
        self.learning_rate = val
