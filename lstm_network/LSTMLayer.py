import numpy as np
from neural_network.NeuralLayer import NeuralLayer as Layer, BiasedNeuralLayer as BiasedLayer

class LSTMLayer(object):
    def __init__(self, in_size, out_size):
        self.forget_gate_layer = BiasedLayer(
            in_size=in_size, # size of input/last output
            out_size=out_size, # size of state
            activation_fn=Layer.activation_sigmoid,
            activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.input_gate_layer = BiasedLayer(
            in_size=in_size, # size of input/last output
            out_size=out_size, # size of update_values_layer (transposed state)
            activation_fn=Layer.activation_sigmoid,
            activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.update_values_layer = BiasedLayer(
            in_size=in_size, # size of input/last output
            out_size=out_size, # size of state
            activation_fn=Layer.activation_tanh,
            activation_fn_deriv=Layer.activation_tanh_deriv)
        self.output_gate_layer = Layer(
            in_size=in_size, # size of input/last output
            out_size=out_size, # size of state
            activation_fn=Layer.activation_sigmoid,
            activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.state = None

    def feed(self, data):
        # apply forget gate
        self.state = np.dot(self.state, self.forget_gate_layer.feed(data))
        # calculate state update values
        update_values = np.dot(
            self.input_gate_layer.feed(data),
            self.update_values_layer.feed(data)
        )
        # apply state update values
        self.state = self.state + update_values
        # calculate output from new state and output gate
        output = np.dot(
            Layer.activation_tanh(self.state),
            self.output_gate_layer.feed(data))
        return output

