import numpy as np
from LSTMLayer import LSTMLayer
from ..neural_network.NeuralLayer import NeuralLayer as Layer


class LSTMNetwork:

    def __init__(self):
        self.layers_spec = None
        self.layers = None
        self.populated = False
        self.trained = False
        self.chars = None
        self.char_to_int = None
        self.int_to_char = None
        self.state_history = None

    def populate(self, char_set, layer_sizes=None):
        layer_sizes = layer_sizes or self.layers_spec or []
        chars = list(char_set)
        if len(chars) == 0:
            raise Exception("first argument char_set is empty!")
        self.chars = chars
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.layers = []
        self.layers.append(LSTMLayer(
            in_size=len(self.chars),
            out_size=layer_sizes[0]))

        if len(layer_sizes) != 0:
            for cur_size in layer_sizes[1:]:
                self.layers.append(LSTMLayer(
                    in_size=self.layers[-1].size,
                    out_size=cur_size))
        self.layers.append(Layer(
            in_size=self.layers[-1].size,
            out_size=len(self.chars),
            activation_fn=Layer.activation_linear,
            activation_fn_deriv=Layer.activation_linear_deriv))
        self.populated = True

    def feedforward(self, inputs):
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        states, results, enc_inputs, enc_outputs = {}, {}, {}, {}
        states[-1] = [layer.state for layer in self.layers[:-1]]  # get initial state of all lstm layers
        for i in xrange(len(inputs)):
            enc_inputs[i] = np.zeros((len(self.chars), 1))  # encode character in 1-of-k representation
            enc_inputs[i][i] = 1
            results[i][-1] = [inputs[i]]

            # feed forward through all layers
            for j in xrange(len(self.layers) - 1):
                results[i][j] = self.layers[j].feed(results[i][j-1]) # get output of each lstm layer
            results[i][len(self.layers)] = self.layers[-1].feed(results[len(self.layers) - 1].T)  # add output of lstm network to results
            states[i] = [layer.state for layer in self.layers[:-1]] # get updated state from all lstm layers

        return results, states

    def backpropagate(self, inputs, targets, results, states):
        deltas = [np.zeros_like(layer.weights) for layer in self.layers]
        deltas.append(0)
        delta_state_next = np.zeros_like(states[0])
        for i in reversed(xrange(len(inputs))):
            deltas[-1] = np.copy(results[i])
            deltas[-1][targets[i]] -= 1
            for j in reversed(xrange(len(deltas) - 1)):
                deltas[j] += np.dot(deltas[j+1], states[i].T)

            for result, layer, last_weights in reversed(xrange(len(self.layers)-1)):
                deltas.append(
                    self.layers[-j-1].get_delta(
                        result[i][-j-1],
                        deltas[-j-1],
                        self.layers.weights[-j-1]
                    )
                )


    def train(self, t_input, seq_length):
        p = 0
        smooth_loss = -np.log(1.0/len(self.chars)) * seq_length
        while True:
            inputs = [self.char_to_int[ch] for ch in t_input[p:p+seq_length]]
            targets = [self.char_to_int[ch] for ch in t_input[1+p:1+p+seq_length]]
            enc_inputs, enc_outputs = {}

            for i in xrange(len(inputs)):
                enc_inputs[i] = np.zeros((len(self.chars), 1))
                enc_inputs[i][inputs[i]] = 1



