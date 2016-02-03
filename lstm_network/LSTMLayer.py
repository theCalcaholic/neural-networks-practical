import os
import numpy as np
from neural_network import \
    NeuralLayer as Layer, \
    BiasedNeuralLayer as BiasedLayer
from LSTMLayerCacheList import LSTMLayerCache
from util import Logger


class LSTMLayer(object):
    def __init__(self, in_size, memory_size):
        concat_size = in_size + memory_size
        self.forget_gate_layer = BiasedLayer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of state
                activation_fn=Layer.activation_sigmoid,
                activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.forget_gate_layer.biases = np.ones(memory_size)
        self.input_gate_layer = BiasedLayer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of update_values_layer (transposed state)
                activation_fn=Layer.activation_sigmoid,
                activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.update_values_layer = BiasedLayer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of state
                activation_fn=Layer.activation_tanh,
                activation_fn_deriv=Layer.activation_tanh_deriv)
        self.output_gate_layer = BiasedLayer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of state
                activation_fn=Layer.activation_sigmoid,
                activation_fn_deriv=Layer.activation_sigmoid_deriv)
        self.pending_updates = LSTMLayerPendingUpdates(in_size, memory_size)
        self.size = memory_size
        self.in_size = in_size
        self.loss_function = LSTMLayer.euclidean_loss_function

        self.first_cache = LSTMLayerCache()
        self.first_cache.is_first_cache = True
        self.first_cache.state = np.zeros((self.size, 1))
        self.first_cache.output_values = np.zeros((self.size, 1))

        self.last_cache = LSTMLayerCache()
        self.first_cache.insert_after(self.last_cache)
        self.last_cache.is_last_cache = True

        self.caches = []

    @classmethod
    def get_convolutional_layer(cls, reference_layer):
        conv_layer = LSTMLayer(
            in_size=reference_layer.in_size,
            memory_size=reference_layer.size
        )
        conv_layer.input_gate_layer = reference_layer.input_gate_layer
        conv_layer.forget_gate_layer = reference_layer.forget_gate_layer
        conv_layer.output_gate_layer = reference_layer.output_gate_layer
        conv_layer.update_values_layer = reference_layer.update_values_layer

        return conv_layer

    def feed(self, input_data, time_steps=1):
        assert time_steps > 0, "time_steps must be at least 1 (for recursive input)!"
        Logger.debug("feed(" + str(input_data) + ")")
        concat_in = np.concatenate([input_data, self.last_cache.predecessor.output_values])
        if time_steps <= len(self.caches):
            self.first_cache.successor.remove()
            self.caches.pop(0)

        self.last_cache.insert_before(LSTMLayerCache())
        cache = self.last_cache.predecessor
        self.caches.append(cache)

        cache.input_values = input_data
        cache.concatenated_input = concat_in
        # update forget gate

        cache.forget_gate_results = self.forget_gate_layer.feed(concat_in)
        cache.input_gate_results = self.input_gate_layer.feed(concat_in)
        cache.update_values_layer_results = self.update_values_layer.feed(concat_in)
        cache.output_gate_results = self.output_gate_layer.feed(concat_in)

        # calculate state update values
        update_values = np.multiply(
                cache.input_gate_results,
                cache.update_values_layer_results)
        # apply forget layer and apply state update values
        cache.state = cache.predecessor.state * cache.forget_gate_results \
                      + update_values
        # calculate output from new state and output gate
        cache.output_values = Layer.activation_tanh(cache.state) * cache.output_gate_results
        # Uncomment for output layer:
        return cache.output_values

    def learn_recursive(self, cache, deltas):
        if len(deltas) == 0 or cache.is_first_cache:
            return
        # calculate loss and cumulative loss but keep loss for t+1 (last loss)
        delta = deltas[-1]

        loss_output = delta + cache.successor.loss_output

        last_loss_state = cache.successor.loss_state

        #print(str(np.shape(cache.output_gate_results)) + str(np.shape(loss_output)) + str(np.shape(last_loss_state)))
        delta_state = cache.output_gate_results * loss_output + last_loss_state

        delta_output_gate = self.output_gate_layer.activation_deriv(
                cache.output_gate_results) * cache.state * loss_output

        delta_input_gate = self.input_gate_layer.activation_deriv(
                cache.input_gate_results) * cache.update_values_layer_results * delta_state

        delta_update_values_layer = self.update_values_layer.activation_deriv(
                cache.update_values_layer_results) * cache.input_gate_results * delta_state

        delta_forget_gate = self.forget_gate_layer.activation_deriv(
                cache.forget_gate_results) * cache.predecessor.state * delta_state

        concat_in = cache.concatenated_input

        self.pending_updates.input_gate_weights += \
            np.outer(delta_input_gate, concat_in)
        self.pending_updates.forget_gate_weights += \
            np.outer(delta_forget_gate, concat_in)
        self.pending_updates.output_gate_weights += \
            np.outer(delta_output_gate, concat_in)
        self.pending_updates.update_values_layer_weights += \
            np.outer(delta_update_values_layer, concat_in)

        self.pending_updates.input_gate_biases += np.ravel(delta_input_gate)
        self.pending_updates.forget_gate_biases += np.ravel(delta_forget_gate)
        self.pending_updates.output_gate_biases += np.ravel(delta_output_gate)
        self.pending_updates.update_values_layer_biases += np.ravel(delta_update_values_layer)

        delta_concatinated_input = np.zeros_like(concat_in) \
                                   + np.dot(self.input_gate_layer.weights.T, delta_input_gate) \
                                   + np.dot(self.forget_gate_layer.weights.T, delta_forget_gate) \
                                   + np.dot(self.output_gate_layer.weights.T, delta_output_gate) \
                                   + np.dot(self.update_values_layer.weights.T, delta_update_values_layer)

        cache.loss_state = delta_state * cache.forget_gate_results
        cache.loss_input = delta_concatinated_input[:self.in_size]
        cache.loss_output = delta_concatinated_input[self.in_size:]

        return self.learn_recursive(cache.predecessor, deltas[:-1])

    def learn(self, deltas, learning_rate=0.001):
        Logger.debug("learn(" + str(deltas) + ")")
        self.learn_recursive(self.last_cache.predecessor, deltas)
        self.apply_training(learning_rate)
        deltas = [cache.loss_input for cache in self.caches]
        return deltas

    def apply_training(self, learning_rate):
        p_updates = self.pending_updates
        lr = learning_rate
        self.forget_gate_layer.weights -= lr * p_updates.forget_gate_weights
        self.input_gate_layer.weights -= lr * p_updates.input_gate_weights
        self.update_values_layer.weights -= lr * p_updates.update_values_layer_weights
        self.output_gate_layer.weights -= lr * p_updates.output_gate_weights
        self.forget_gate_layer.biases -= lr * p_updates.forget_gate_biases
        self.input_gate_layer.biases -= lr * p_updates.input_gate_biases
        self.update_values_layer.biases -= lr * p_updates.update_values_layer_biases
        self.output_gate_layer.biases -= lr * p_updates.output_gate_biases
        p_updates.reset()

        for matrix in [
                self.forget_gate_layer.weights,
                self.input_gate_layer.weights,
                self.update_values_layer.weights,
                self.output_gate_layer.weights,
                self.forget_gate_layer.biases,
                self.input_gate_layer.biases,
                self.update_values_layer.biases,
                self.output_gate_layer.biases]:
            np.clip(matrix, -5, 5, out=matrix)

    def save(self, directory):
        self.forget_gate_layer.save(os.path.join(directory, "forget_gate.npz"))
        self.input_gate_layer.save(os.path.join(directory, "input_gate.npz"))
        self.output_gate_layer.save(os.path.join(directory, "output_gate.npz"))
        self.update_values_layer.save(os.path.join(directory, "update_values_layer.npz"))

    def load(self, directory):
        self.forget_gate_layer.load(os.path.join(directory, "forget_gate.npz"))
        self.input_gate_layer.load(os.path.join(directory, "input_gate.npz"))
        self.output_gate_layer.load(os.path.join(directory, "output_gate.npz"))
        self.update_values_layer.load(os.path.join(directory, "update_values_layer.npz"))

    @classmethod
    def get_output_error(cls, prediction, target):
        difference = np.zeros_like(prediction)
        tmp = 2 * np.sum(prediction - target)
        difference = tmp
        return difference

    @classmethod
    def euclidean_loss_function(cls, predicted, target):
        return (predicted - target) ** 2

    def visualize(self, path, layer_id):
        self.input_gate_layer.visualize(os.path.join(path, "LSTM" + str(layer_id), "obs_" + "InputG_" + str(layer_id) + "_0.pgm"))
        self.forget_gate_layer.visualize(os.path.join(path, "LSTM" + str(layer_id), "obs_" + "ForgetG_" + str(layer_id * 2) + "_0.pgm"))
        self.output_gate_layer.visualize(os.path.join(path, "LSTM" + str(layer_id), "obs_" + "OutputG_" + str(layer_id * 3) + "_0.pgm"))
        self.update_values_layer.visualize(os.path.join(path, "LSTM" + str(layer_id), "obs_" + "UpdateL_" + str(layer_id * 4) + "_0.pgm"))

    def clear_cache(self):
        self.caches = []
        self.first_cache.successor = self.last_cache
        self.last_cache.predecessor = self.first_cache


class LSTMLayerPendingUpdates(object):
    def __init__(self, in_size, mem_size):
        self.update_values_layer_weights = np.zeros((mem_size, in_size + mem_size))
        self.input_gate_weights = np.zeros((mem_size, in_size + mem_size))
        self.forget_gate_weights = np.zeros((mem_size, in_size + mem_size))
        self.output_gate_weights = np.zeros((mem_size, in_size + mem_size))
        self.update_values_layer_biases = np.zeros(mem_size)
        self.input_gate_biases = np.zeros(mem_size)
        self.forget_gate_biases = np.zeros(mem_size)
        self.output_gate_biases = np.zeros(mem_size)
        self.in_size = in_size
        self.mem_size = mem_size

    def reset(self):
        self.__init__(self.in_size, self.mem_size)

    @classmethod
    def activation_fn_forget_gate(cls, x):
        return Layer.activation_sigmoid(x)

