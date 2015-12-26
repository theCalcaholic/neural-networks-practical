import numpy as np
from ..neural_network.NeuralLayer import \
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
        self.cache = [LSTMLayerCache(out_size)]
        self.deltas = LSTMLayerDeltas(out_size, in_size + out_size)
        self.size = out_size
        self.in_size = in_size
        self.learning_rate = 0.001

    def feed(self, input_data, caching_depth=1):
        assert caching_depth > 0, "caching_depth must be at least 1 (for recursive input)!"
        ##print("ff input shape: " + str(np.shape(input_data)) + " - last output shape: " + str(np.shape(self.cache[-1].output_values)))
        data = np.concatenate([input_data, self.cache[-1].output_values])
        if caching_depth <= len(self.cache):
            self.cache.pop(0)
        self.cache.append(LSTMLayerCache(self.size))
        self.cache[-1].input_values = input_data
        self.cache[-1].input_concat = data
        # update forget gate
        self.cache[-1].forget_gate = self.forget_gate_layer.feed(data)
        self.cache[-1].input_gate = self.input_gate_layer.feed(data)
        self.cache[-1].update_values = self.update_values_layer.feed(data)
        self.cache[-1].output_gate = self.output_gate_layer.feed(data)
        # calculate state update values
        ##print("ff update values shape: " + str(np.shape(self.cache[-1].update_values)) + " - input gate shape: " + str(np.shape(self.cache[-1].input_gate)))
        self.cache[-1].update_values = np.multiply(
            self.cache[-1].input_gate,
            self.cache[-1].update_values)
        # apply forget layer
        ##print("ff forget gate: " + str(self.cache[-1].forget_gate))
        self.cache[-1].state = self.cache[-1].state * self.cache[-1].forget_gate
        ##print("ff forgotten state shape: " + str(np.shape(self.cache[-1].state)))
        # apply state update values
        self.cache[-1].state = self.cache[-1].state + self.cache[-1].update_values
        ##print("ff updated state shape: " + str(np.shape(self.cache[-1].state)))
        ##print("ff output values shape: " + str(np.shape(self.cache[-1].output_gate)))
        # calculate output from new state and output gate
        self.cache[-1].output_values = Layer.activation_tanh(self.cache[-1].state) * self.cache[-1].output_gate
        ##print("ff output shape: " + str(np.shape(self.cache[-1].output_values)) + "\nff " + str(self.cache[-1].output_values))
        return self.cache[-1].output_values

    def learn(self, target_output):
        print("output_values shape: " + str(np.shape(self.cache[-1].output_values)) + " - target shape: " + str(np.shape(target_output)))
        incremental_error = LSTMLayer.get_error(self.cache[-1].output_values, target_output)
        output_error = LSTMLayer.get_output_error(self.cache[-1].output_values, target_output)
        state_error = np.zeros(self.size)
        while len(self.cache) > 0:
            cache = self.cache.pop()
            delta_state = cache.output_gate * output_error + state_error
            delta_output_gate = cache.state * output_error
            delta_input_gate = cache.update_values * delta_state
            delta_update_layer = cache.input_gate * delta_state
            delta_forget_gate = self.cache[-1].state * delta_state

            # deltas with respective to vector inside activation functions
            delta_input_gate_act = (1. - cache.input_gate) * cache.input_gate * delta_input_gate
            delta_forget_gate_act = (1. - cache.forget_gate) * cache.forget_gate * delta_forget_gate
            delta_output_gate_act = (1. - cache.output_gate) * cache.output_gate * delta_output_gate
            delta_update_layer_act = (1. - cache.update_values ** 2) * delta_update_layer

            # update pending deltas
            print("delta_input_gate_act shape: " + str(np.shape(delta_input_gate_act)) + " - input_concat shape: " + str(np.shape(cache.input_concat)))
            test = np.outer(delta_input_gate_act, cache.input_concat)
            print("test shape: " + str(np.shape(test)))
            self.deltas.input_gate_weights += test
            self.deltas.forget_gate_weights += np.outer(delta_forget_gate_act, cache.input_concat)
            self.deltas.output_gate_weights += np.outer(delta_output_gate_act, cache.input_concat)
            self.deltas.update_values_weights += np.outer(delta_update_layer_act, cache.input_concat)
            self.deltas.input_gate_biases += delta_input_gate_act
            self.deltas.forget_gate_biases += delta_forget_gate_act
            self.deltas.output_gate_biases += delta_output_gate_act
            self.deltas.update_values_biases += delta_update_layer_act

            delta_x_concat = np.zeros_like(cache.input_concat)
            delta_x_concat += np.dot(self.input_gate_layer.weights.T, delta_input_gate_act)
            delta_x_concat += np.dot(self.forget_gate_layer.weights.T, delta_forget_gate_act)
            delta_x_concat += np.dot(self.output_gate_layer.weights.T, delta_output_gate_act)
            delta_x_concat += np.dot(self.update_values_layer.weights.T, delta_update_layer_act)

            difference_state = delta_state * cache.forget_gate
            difference_input = delta_x_concat[:self.in_size]
            difference_output = delta_x_concat[self.in_size:]

            incremental_error = LSTMLayer.get_error(self.cache[-1].output_values, cache.input_values)
            output_error = LSTMLayer.get_output_error(self.cache[-1].output_values, cache.input_values)
            output_error += difference_output
            state_error = difference_state

        self.apply_deltas()
        return incremental_error

    def apply_deltas(self):
        self.update_values_layer.weights -= self.learning_rate * self.deltas.update_values_weights
        self.input_gate_layer.weights -= self.learning_rate * self.deltas.input_gate_weights
        self.forget_gate_layer.weights -= self.learning_rate * self.deltas.forget_gate_weights
        self.output_gate_layer.weights -= self.learning_rate * self.deltas.output_gate_weights
        self.update_values_layer.biases -= self.learning_rate * self.deltas.update_values_biases
        self.input_gate_layer.biases -= self.learning_rate * self.deltas.input_gate_biases
        self.forget_gate_layer.biases -= self.learning_rate * self.deltas.forget_gate_biases
        self.output_gate_layer.biases -= self.learning_rate * self.deltas.output_gate_biases
        self.deltas = LSTMLayerDeltas()

    @classmethod
    def get_output_error(cls, prediction, target):
        difference = np.zeros_like(prediction)
        tmp = 2 * np.sum(prediction - target)
        difference = tmp
        return difference

    @classmethod
    def get_error(cls, prediction, target):
        return (prediction - target) ** 2


class LSTMLayerCache(object):
    def __init__(self, out_size):
        self.update_values = np.zeros((out_size, 1))
        self.input_gate = np.zeros(out_size)
        self.forget_gate = np.zeros(out_size)
        self.output_gate = np.zeros(out_size)
        self.state = np.zeros((out_size, 1))
        self.output_values = np.zeros((out_size, 1))
        self.input_values = None
        self.input_concat = None


class LSTMLayerDeltas(object):
    def __init__(self, in_size, out_size):
        self.update_values_weights = np.zeros((out_size, in_size + out_size))
        self.input_gate_weights = np.zeros((out_size, in_size + out_size))
        self.forget_gate_weights = np.zeros((out_size, in_size + out_size))
        self.output_gate_weights = np.zeros((out_size, in_size + out_size))
        self.update_values_biases = np.zeros(out_size)
        self.input_gate_biases = np.zeros(out_size)
        self.forget_gate_biases = np.zeros(out_size)
        self.output_gate_biases = np.zeros(out_size)