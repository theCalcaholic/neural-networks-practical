import numpy as np
from LSTMLayer import LSTMLayer
from utils import decode, decode_sequence
import math


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
        self.verbose = True

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

        outputs = []
        # get initial state of all lstm layers
        for t in xrange(len(inputs)):
            # feed forward through all layers
            # get output of lstm network
            outputs.append(self.lstm_layer.feed(inputs[t], caching_depth))
            # get updated state from all lstm layers

        return outputs

    def backpropagate(self, targets_list):
        self.output_layer.get_delta(
            self.last_cache.predecessor.output_layer_results,
        )
        return self.lstm_layer.learn(targets_list)

    def train(self, inputs_list, seq_length, iterations=100, learning_rate=0.1):
        #print("inputs_list: " + str(inputs_list))
        loss = 1.0
        loss_diff = 0
        limit = 50
        for i in xrange(iterations):
            pos = 0
            output_string = ""
            input_string = ""
            for inputs, targets in zip(
                    [inputs_list[j:j+seq_length] for j in xrange(0, len(inputs_list) - 1, seq_length)],
                    [inputs_list[j:j+seq_length] for j in xrange(1, len(inputs_list), seq_length)]):
                outputs = self.feedforward(inputs, seq_length)
                loss = self.lstm_layer.learn(targets, learning_rate)
                pos += seq_length

                output_string += decode_sequence(outputs, self.int_to_data)
                input_string += decode_sequence(inputs, self.int_to_data)
            loss = np.average(loss)

            if self.verbose and not i % 10:
                loss_diff -= loss
                print("\nIteration " + str(i) + " - learning rate: " + str(learning_rate) + "  ")
                      #+ ("\\/" if learning_rate_diff > 0 else ("--" if learning_rate_diff == 0 else "/\\"))
                      #                                        + " " + str(learning_rate_diff)[1:])
                print("loss: " + str(loss) + "  "
                      + ("\\/" if loss_diff > 0 else ("--" if loss_diff == 0 else "/\\"))
                      + " " + str(loss_diff)[1:])
                print("in: " + input_string)
                print("out: " + output_string)
                if loss_diff < 0:
                    learning_rate /= 1.2
                elif loss_diff < (loss / limit):
                    learning_rate *= 1.2
                    limit += 2
                loss_diff = loss

    def predict(self, inputs, length=1):
        results = [inputs]
        for i in range(length):
            results.append(self.feedforward([results[-1]]))
        return results
