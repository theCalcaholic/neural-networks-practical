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
        self.learning_rate = 0.001

    def populate(self, char_set, layer_sizes=None):
        layer_sizes = layer_sizes or self.layers_spec or []
        chars = list(char_set)
        if len(chars) == 0:
            raise Exception("first argument char_set is empty!")
        self.chars = chars

        self.layers = []

        self.lstm_layer = LSTMLayer(
            in_size=len(self.chars),
            memory_size=layer_sizes[0])
        """self.output_layer = Layer(
            in_size=self.lstm_layer.size,
            out_size=len(self.chars))

        self.lstm_layer.output_layer = self.output_layer
            #out_size=len(self.chars))"""


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
        self.output_layer.get_delta(
            self.last_cache.predecessor.output_layer_results,
        )
        return self.lstm_layer.learn(targets_list)

    def train(self, inputs_list, seq_length, iterations=100, learning_rate=0.001):
        #print("inputs_list: " + str(inputs_list))
        output_strings = "          "
        for i in xrange(iterations):
            pos = 0
            while pos < len(inputs_list):
                if pos + seq_length < len(inputs_list) - 1:
                    inputs = inputs_list[pos: pos + seq_length]
                    targets = inputs_list[pos + 1: pos + seq_length + 1]
                else:
                    inputs = inputs_list[pos: -1]
                    targets = inputs_list[pos + 1:]
                #print("in_length: " + str(len(inputs)))
                #print("t_length: " + str(len(targets)))
                #print("targets: " + str(targets))
                output = self.feedforward(inputs, seq_length)
                loss = self.lstm_layer.learn(targets, learning_rate)
                pos += seq_length
                output_strings = output_strings[1:] + decode(output, self.int_to_data)
            loss = np.average(loss)
            print("Iteration " + str(i))
            print("loss: " + str(loss))
            print("out: " + output_strings)

    def predict(self, inputs, length=1):
        results = [inputs]
        for i in range(length):
            results.append(self.feedforward([results[-1]]))
        return results
