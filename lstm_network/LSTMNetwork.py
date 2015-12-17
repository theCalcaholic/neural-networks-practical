import numpy as np
from LSTMLayer import LSTMLayer
from neural_network.NeuralLayer import NeuralLayer as Layer


class LSTMNetwork:
    def __init__(self):
        self.layers_spec = None
        self.layers = None
        self.populated = False
        self.trained = False

    def populate(self, in_size, layer_sizes=None):
        layer_sizes = layer_sizes or self.layers_spec or []
        self.layers = []
        self.layers.append(LSTMLayer(
            in_size=in_size,
            out_size=layer_sizes[0]))

        if len(layer_sizes) != 0:
            for cur_size in layer_sizes[1:-1]:
                self.layers.append(LSTMLayer(
                    in_size=self.layers[-1].size,
                    out_size=cur_size))
        self.layers.append(Layer(
            in_size=self.layers[-1].size,
            out_size=layer_sizes[-1],
            activation_fn=Layer.activation_linear,
            activation_fn_deriv=Layer.activation_linear_deriv))
        self.populated = True

    def feedforward(self, data):
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        results = [data]
        for layer in self.layers[:-1]:
            results.append(
                layer.feed(results[-1])
            )
        results.append(
            self.layers[-1].feed(results[-1].T)
        )

        return results

