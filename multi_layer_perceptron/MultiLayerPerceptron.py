import numpy

from ..neural_network.NeuralLayer import NeuralLayer as Layer, BiasedNeuralLayer as BiasedLayer

# directory for observing the results
path_obs = "res/"


class MultiLayerPerceptron:

    def __init__(self, in_size=0, layers=None, config=dict()):
        """Creates a new instance of class MLP
        @size_in: number of input neurons
        @size_out: number of output neurons
        @size_hid: number of hidden layer neurons"""

        assert isinstance(config, dict)
        self.populated = False
        self.trained = False
        self.layers = []

        self.layers_specs = layers
        self.config = {
            "learning_rate": 0.02,
            "target_precision": 0,
            "max_iterations": 10000
        }
        self.configure(config)

        if in_size and len(layers) >= 0:
            self.populate(in_size, layers)

    def populate(self, in_size,
                 layers=None):
        layers = layers or self.layers_specs or []
        self.layers = []
        self.layers.append(BiasedLayer(
            in_size=in_size,
            out_size=layers[0]["size"],
            activation_fn=layers[0]["fn"] or Layer.activation_sigmoid,
            activation_fn_deriv=layers[0]["fn_deriv"] or Layer.activation_sigmoid_deriv))

        if len(layers) != 0:
            for cur_layer in layers[1:-1]:
                self.layers.append(BiasedLayer(
                    in_size=self.layers[-1].size,
                    out_size=cur_layer["size"],
                    activation_fn=cur_layer["fn"] or Layer.activation_sigmoid,
                    activation_fn_deriv=cur_layer["fn_deriv"] or Layer.activation_sigmoid_deriv))

        self.layers.append(Layer(
            in_size=self.layers[-1].size,
            out_size=layers[-1]["size"],
            activation_fn=Layer.activation_linear,
            activation_fn_deriv=Layer.activation_linear_deriv))
        self.populated = True

    def feedforward(self, data):
        """applies the MLP network to a set of input data and returns a list of the outputs of each perceptron layer"""

        if not self.populated:
            raise Exception("MLP must be populated first (Have a look at method MLP.populate())!")

        results = [data]
        for layer in self.layers:
            results.append(
                layer.feed(results[-1])
            )

        return results

    def backpropagate(self, results, target_output):
        """calculates and applies the change in weight according to the error resulting from feedforward and the
        learning rate provided
        returns the absolute error
        @results: An array of layer wise results (as produced by feedforward)
        @target_output: A vector of output values to be used as training reference
        @learning_rate: Influences the performance (the higher the better) and accuracy (the lower the better)"""

        if not self.populated:
            raise Exception("MLP must be populated first (Have a look at method MLP.populate())!")

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
            layer.learn(result, delta, self.config["learning_rate"])

        # returns the absolute error (distance of target output and actual output)
        return error

    def train(self, t_input, t_output):
        """Trains the MLP with the given training data
        @t_input: Set of training input vectors
        @t_output: Set of training output vectors matching t_input
        @iterations: number of training epochs
        @learning_rate: Influences the performance (the higher the better) and accuracy (the lower the better)"""

        t_input = t_input
        for i in range(self.config["max_iterations"]):
            errors = []
            for j in range(0, numpy.shape(t_input)[0]):
                errors.append(self.backpropagate(self.feedforward(t_input[j]), t_output[j]))
            error = sum(errors) / float(len(errors))

            if not i % 100:
                print "Error: ", error, "\n"

            if error < self.config["target_precision"]:
                self.trained = True
                print "Target error reached: ", str(error), " < " + str(self.config["target_precision"]) + "\n"
                return
        self.trained = True

    def predict(self, data_in):
        """Returns predicted output vector for a given input vector data_in"""

        if not self.trained:
            raise Exception("MLP must be trained first!")
        return self.feedforward(data_in)[-1][0]

    def print_layers(self):
        for i in range(len(self.layers)):
            print("Layer " + str(i) + ":")
            print("  shape: " + str(numpy.shape(self.layers[i].weights)))
            print("  activation_fn: " + str(self.layers[i].activation))

    def add_layer(self, size, fn, fn_deriv):
        """Adds a hidden layer to the MLP.
        @size: the number of nodes the new layer will have
        @fn: the layers activation function
        @fn_deriv: the layers activation functions derivation"""

        self.layers_specs.append({
            "size": size,
            "fn": fn,
            "fn_deriv": fn_deriv
        })

    def add_layers(self, layer_list):
        """Adds a list of hidden layers to the MLP. Each layer must be a dict containing
            the keys 'size', 'fn' and 'fn_deriv'"""

        for layer in layer_list:
            self.add_layer(layer["size"], layer["fn"], layer["fn_deriv"])

    def configure(self, config):
        """Expects a dictionary of configuration options to update the existing values. Possible options are:
        learning_rate (float): Determines how fast and precise the network is going to learn -
                higher learning rate means faster learning and lower learning precision.
        target_precision (float): Sets the target precision for early stopping. To disable early stopping at all,
                set this value to 0.
        max_iterations (float): The maximum number of iterations over which the MLP should be trained"""

        assert isinstance(config, dict)
        assert "learning_rate" not in config or isinstance(config["learning_rate"], float) \
            and config["max_iterations"] > 0
        assert "target_precision" not in config or isinstance(config["target_precision"], float)
        assert "max_iterations" not in config or isinstance(config["max_iterations"], int) \
            and config["max_iterations"] > 0
        self.config.update(config)
