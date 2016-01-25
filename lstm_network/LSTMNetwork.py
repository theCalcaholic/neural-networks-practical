import os
import numpy as np
from LSTMLayer import LSTMLayer
from utils import encode, decode, decode_sequence
import math


class LSTMNetwork:

    def __init__(self):
        self.layers_spec = None
        self.layers = None
        self.populated = False
        self.trained = False
        self.chars = None
        self._int_to_data = None
        self._data_to_int = None
        self.state_history = None
        self.learning_rate = 0.001
        self.verbose = True
        self.lstm_layer = None
        self.lstm_layer2 = None

    def populate(self, char_set, layer_sizes=None):
        layer_sizes = layer_sizes or self.layers_spec or []
        chars = list(char_set)
        if len(chars) == 0:
            raise Exception("first argument char_set is empty!")
        self.chars = chars

        self.layers = []

        self.lstm_layer = LSTMLayer(
            in_size=len(self.chars),
            out_size=len(self.chars),
            memory_size=layer_sizes[0])
        self.lstm_layer2 = LSTMLayer(
            in_size=len(self.chars),
            out_size=len(self.chars),
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

    def train(self, inputs_list, seq_length, iterations=100, learning_rate=0.1, save_dir="lstm_save", iteration_start=0):
        #print("inputs_list: " + str(inputs_list))
        loss = 1.0
        loss_diff = 0
        limit = 30
        av_loss_diff = 0
        for i in xrange(iteration_start, iterations):
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

            if not i % 10:
                self.lstm_layer.save(os.path.join(os.getcwd(), save_dir))
                loss_diff -= loss
                if self.verbose:
                    print("\nIteration " + str(i) + " - learning rate: " + str(learning_rate) + "  ")
                          #+ ("\\/" if learning_rate_diff > 0 else ("--" if learning_rate_diff == 0 else "/\\"))
                          #                                        + " " + str(learning_rate_diff)[1:])
                    print("loss: " + str(loss) + "  "
                          + ("\\/" if loss_diff > 0 else ("--" if loss_diff == 0 else "/\\"))
                          + " " + str(loss_diff)[1:])
                    print("in: " + input_string.replace("\n", "\\n"))
                    print("out: " + output_string.replace("\n", "\\n"))
                    av_loss_diff += abs(loss_diff)
                    if i == 0:
                        av_loss_diff = 0
                    if i > 0:
                        print("average lossdiff: " + str(av_loss_diff / i))
                    if not i % 100:
                        self.lstm_layer2.load(save_dir)
                        free = "a"
                        for i in range(30):
                            free_in = encode(ord(free[-1]), self.data_to_int, len(self.chars))
                            free += decode(self.lstm_layer2.feed(free_in), self.int_to_data)
                        print("freestyle: " + free)

                #math.sqrt(loss) / 10
                """if loss > 1:
                    self.roll_weights()
                    i = -1"""

                """if loss_diff < 0:
                    learning_rate /= 1.2
                elif loss_diff < (loss / limit):
                    learning_rate *= 1.2
                    limit += 1"""
                loss_diff = loss

    def roll_weights(self):
        self.populate(self.chars, [self.lstm_layer.size])

    def predict(self, inputs, length=1):
        results = [inputs]
        for i in range(length):
            results.append(self.feedforward([results[-1]]))
        return results

    def load(self, path):
        self.lstm_layer.load(os.path.join(os.getcwd(), path))


    def data_to_int(self, x):
        if x in self._data_to_int:
            return self._data_to_int[x]
        else:
            return -1


    def int_to_data(self, x):
        return "#" if self._int_to_data[x] > 255 else chr(self._int_to_data[x])
