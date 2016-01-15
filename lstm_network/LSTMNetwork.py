import numpy as np
from LSTMLayer import LSTMLayer
from neural_network.NeuralLayer import NeuralLayer as Layer
from utils import decode

class LSTMNetwork:

    def __init__(self):
        self.layers_spec = None
        self.layers = None
        self.populated = False
        self.trained = False
        self.chars = None
        self.int_to_data = None
        self.state_history = None

    def populate(self, char_set, layer_sizes=None):
        layer_sizes = layer_sizes or self.layers_spec or []
        chars = list(char_set)
        if len(chars) == 0:
            raise Exception("first argument char_set is empty!")
        self.chars = chars

        self.layers = []
        self.lstm_layer = LSTMLayer(
            in_size=len(self.chars),
            out_size=layer_sizes[0])

        """self.lstm_layer = LSTMLayer(
            in_size=layer_sizes[0],
            out_size=layer_sizes[1])"""

        """if len(layer_sizes) != 0:
            for cur_size in layer_sizes[1:]:
                self.layers.append(LSTMLayer(
                    in_size=self.layers[-1].size,
                    out_size=cur_size))"""
        """self.layers.append(Layer(
            in_size=self.layers[-1].size,
            out_size=len(self.chars),
            activation_fn=Layer.activation_linear,
            activation_fn_deriv=Layer.activation_linear_deriv))
        self.layers[-1].weights.fill(1.0)
        print("last layer shape: " + str(np.shape(self.layers[-1].weights)))"""
        self.populated = True

    def feedforward(self, inputs, caching_depth=1):
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        outputs = None
        # get initial state of all lstm layers
        for t in xrange(len(inputs)):
            # feed forward through all layers
            # get output of lstm network
            outputs = self.lstm_layer.feed(inputs[t], caching_depth)
            # get updated state from all lstm layers

        return outputs

    def backpropagate(self, targets_list):
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
        """print("enc_target shape: " + str(np.shape(enc_target.T)) + " - delta shape: " + str(np.shape(delta.T))
              + " - weights shape: " + str(np.shape(self.layers[-1].weights))
              + " - biases shape: " + str(np.shape(np.atleast_2d(self.layers[-1].biases))))"""
        self.lstm_layer.learn(targets_list)

    def train(self, inputs_list, seq_length):
        for i in range(100):
            pos = 0
            while pos < len(inputs_list):
                if pos + seq_length < len(inputs_list) - 1:
                    inputs = inputs_list[pos: pos + seq_length]
                    targets = inputs_list[pos + 1: pos + seq_length + 1]
                else:
                    print("else")
                    inputs = inputs_list[pos: -1]
                    targets = inputs_list[pos + 1:]
                print("in_length: " + str(len(inputs)))
                print("t_length: " + str(len(targets)))
                output = self.feedforward(inputs)
                self.lstm_layer.learn(targets)
                pos += seq_length
                print("Iteration " + str(i))
                print(decode(output, self.int_to_data))


