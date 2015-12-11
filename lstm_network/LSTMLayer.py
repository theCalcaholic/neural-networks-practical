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
        self.state = np.random.uniform(0.0, 1.0, (out_size))
        self.last_output = np.zeros(out_size)
        self.size = out_size

    def feed(self, input_data):
        data = np.concatenate((input_data, self.last_output))
        ##print("input_size + output_size:" + str(np.shape(data)))
        ##print("size: " + str(self.size))
        ##print("state shape: " + str(np.shape(self.state)))
        # update forget gate
        forget_gate = self.forget_gate_layer.feed(data)
        ##print("forget gate shape: " + str(np.shape(forget_gate)))
        # calculate state update values
        update_values = np.inner(
            self.input_gate_layer.feed(data),
            self.update_values_layer.feed(data))
        # apply forget layer
        ##print("forget gate: " + str(forget_gate))
        self.state = self.state * forget_gate
        ##print("forgotten state shape: " + str(np.shape(self.state)))
        # apply state update values
        self.state = self.state + update_values
        ##print("updated state shape: " + str(np.shape(self.state)))
        ##print("output gate shape: " + str(np.shape(self.output_gate_layer.feed(data))))
        # calculate output from new state and output gate
        output_gate = self.output_gate_layer.feed(data)
        self.last_output = Layer.activation_tanh(self.state) * output_gate
        ##print("output shape: " + str(np.shape(self.last_output)) + "\n" + str(self.last_output))
        return self.last_output


    def learn(self):
        pass