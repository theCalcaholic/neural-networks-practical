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
        """self.layers.append(Layer(
            in_size=self.layers[-1].size,
            out_size=len(self.chars),
            activation_fn=Layer.activation_linear,
            activation_fn_deriv=Layer.activation_linear_deriv))
        self.layers[-1].weights.fill(1.0)
        print("last layer shape: " + str(np.shape(self.layers[-1].weights)))"""
        self.populated = True

    def feedforward(self, inputs):
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        results, enc_inputs, enc_outputs = [], {}, {}
        # get initial state of all lstm layers
        for t in xrange(len(inputs)):
            # encode character in 1-of-k representation
            enc_inputs[t] = np.zeros((len(self.chars), 1))
            enc_inputs[t][inputs[t]] = 1
            results.append([enc_inputs[t]])

            # feed forward through all layers
            for l in xrange(len(self.layers)-1):
                # get output of each lstm layer
                results[t].append(self.layers[l].feed(results[t][-1]))
            # get updated state from all lstm layers

        return results

    def backpropagate(self, results, target):
        """deltas = [np.zeros_like(layer.weights) for layer in self.layers]
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
                )"""
        #print("results shape: " + str(np.shape(results[-1][-1])))
        enc_target = self.encode(target)
        delta = enc_target - results[-1]
        """print("enc_target shape: " + str(np.shape(enc_target.T)) + " - delta shape: " + str(np.shape(delta.T))
              + " - weights shape: " + str(np.shape(self.layers[-1].weights))
              + " - biases shape: " + str(np.shape(np.atleast_2d(self.layers[-1].biases))))"""
        self.layers[-1].learn(delta.T)

    def train(self, t_input, seq_length):
        for i in range(100):
            p = 0
            while p < len(t_input):
                if p + seq_length < len(t_input) - 1:
                    inputs = [self.char_to_int[ch] for ch in t_input[p:p+seq_length]]
                    target = self.char_to_int[t_input[p + seq_length + 1]]
                else:
                    inputs = [self.char_to_int[ch] for ch in t_input[p:-2]]
                    target = self.char_to_int[t_input[-1]]
                output = self.feedforward(inputs)
                self.backpropagate(output, target)
                p += seq_length
                print("Iteration " + self.int_to_char[i])

    def encode(self, input):
        result = np.zeros(len(self.chars))
        result[input] = 1
        return result

    def decode(self, output):
        for index, value in enumerate(output):
            if value == 1:
                return index


