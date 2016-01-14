from NeuralLayer import NeuralLayer, BiasedNeuralLayer
import numpy as np


class ElmanHiddenLayer(BiasedNeuralLayer):

    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        self.context_layer = BiasedNeuralLayer(
                in_size=out_size,
                out_size=out_size,
                activation_fn=NeuralLayer.activation_sigmoid,
                activation_fn_deriv=NeuralLayer.activation_sigmoid_deriv)
        self.context = np.zeros((out_size, 1))
        super(self.__class__).__init__(in_size + out_size, out_size, activation_fn, activation_fn_deriv)

    def feed(self, data):
        output = super(self.__class__).feed(
                np.vstack((
                    data,
                    self.context)))
        self.context = self.context_layer.feed(output)
        return output

    def learn(self, result, delta, learning_rate):
        # TODO: Backpropagation over time
        super(self.__class__).learn(result="", delta="", learning_rate="")