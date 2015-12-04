import numpy as np
from neural_network.NeuralLayer import \
    NeuralLayer as Layer, \
    BiasedNeuralLayer as BiasedLayer


class LSTMLayer(object):
    def __init__(self, in_size, out_size):
        self.forget_gate_layer = BiasedLayer(
            in_size=in_size + out_size,  # size of input/last output
            out_size=out_size,  # size of state
            activation_fn=Layer.activation_sigmoid,
            activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.input_gate_layer = BiasedLayer(
            in_size=in_size + out_size,  # size of input/last output
            out_size=out_size,  # size of update_values_layer (transposed state)
            activation_fn=Layer.activation_sigmoid,
            activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.update_values_layer = BiasedLayer(
            in_size=in_size + out_size,  # size of input/last output
            out_size=out_size,  # size of state
            activation_fn=Layer.activation_tanh,
            activation_fn_deriv=Layer.activation_tanh_deriv)
        self.output_gate_layer = Layer(
            in_size=in_size + out_size,  # size of input/last output
            out_size=out_size,  # size of state
            activation_fn=Layer.activation_sigmoid,
            activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.state = np.random.uniform(0.0, 1.0, (out_size, in_size + out_size))
        self.last_output = np.zeros(out_size)
        self.size = out_size

    def feed(self, data):
        print("size: " + str(self.size))
        print("state shape: " + str(np.shape(self.state)))
        # apply forget gate
        forget_gate = self.forget_gate_layer.feed(data)
        print("forget gate shape: " + str(np.shape(forget_gate)))
        self.state = np.dot(np.atleast_2d(forget_gate), self.state)
        # calculate state update values
        update_values = np.dot(
            self.input_gate_layer.feed(data),
            self.update_values_layer.feed(data)
        )
        # apply state update values
        print("forgotten state shape: " + str(np.shape(self.state)))
        self.state = self.state + update_values
        print("updated state shape: " + str(np.shape(self.state)))
        print("output gate shape: " + str(np.shape(self.output_gate_layer.feed(data))))
        # calculate output from new state and output gate
        self.last_output = np.dot(
            np.atleast_2d(
                Layer.activation_tanh(self.state)).T,
            np.atleast_2d(
                self.output_gate_layer.feed(data)))
        print("output shape: " + str(np.shape(self.last_output)) + "\n" + str(self.last_output))
        return self.last_output

