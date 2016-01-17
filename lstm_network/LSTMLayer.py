import numpy as np
from neural_network.NeuralLayer import \
    NeuralLayer as Layer, \
    BiasedNeuralLayer as BiasedLayer
from LSTMLayerCacheList import LSTMLayerCache
from utils import debug

import pycuda.autoinit
import pycuda.gpuarray
import pycuda.driver
import scuda.linalg
import scuda.cumisc

culinalg.init()

class LSTMLayer(object):
    def __init__(self, in_size, memory_size):
        concat_size = in_size + memory_size
        self.forget_gate_layer = BiasedLayer(
            in_size=concat_size,  # size of input/last output
            out_size=memory_size,  # size of state
            activation_fn=Layer.activation_sigmoid,
            activation_fn_deriv=Layer.activation_sigmoid_deriv)
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
        self.output_gate_layer = Layer(
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

    def feed(self, input_data, caching_depth=1):
        assert caching_depth > 0, "caching_depth must be at least 1 (for recursive input)!"
        debug("feed(" + str(input_data) + ")")
        debug("caching_depth: " + str(len(self.caches)) + "/" + str(caching_depth))
        #print("ff input shape: " + str(np.shape(input_data)) + " - last output shape: " + str(np.shape(self.last_cache.predecessor.output_values)))
        data = np.concatenate([input_data, self.last_cache.predecessor.output_values])
        if caching_depth <= len(self.caches):
            self.first_cache.successor.remove()
            self.caches.pop(0)
        """self.caches.append(LSTMLayerCache())
        cache = self.caches[-1]
        if len(self.caches) == 1:
            cache.predecessor = self.first_cache
            self.first_cache.successor = cache

        cache.predecessor = self.last_cache.predecessor
        self.last_cache.predecessor = cache
        cache.successor = self.last_cache"""

        self.last_cache.insert_before(LSTMLayerCache())
        cache = self.last_cache.predecessor
        self.caches.append(cache)

        cache.input_values = input_data
        cache.concatenated_input = data
        # update forget gate
        cache.forget_gate_results = self.forget_gate_layer.feed(data)
        cache.input_gate_results = self.input_gate_layer.feed(data)
        cache.update_values_layer_results = self.update_values_layer.feed(data)
        cache.output_gate_results = self.output_gate_layer.feed(data)
        # calculate state update values
        ##print("ff update values shape: " + str(np.shape(self.cache[-1].update_values)) + " - input gate shape: " + str(np.shape(self.cache[-1].input_gate)))
        update_values = np.multiply(
            cache.input_gate_results,
            cache.update_values_layer_results)
        # apply forget layer and apply state update values
        ##print("ff forget gate: " + str(self.cache[-1].forget_gate))
        cache.state = cache.predecessor.state * self.last_cache.predecessor.forget_gate_results \
                      + update_values
        ##print("ff forgotten state shape: " + str(np.shape(self.cache[-1].state)))
        ##print("ff updated state shape: " + str(np.shape(self.cache[-1].state)))
        ##print("ff output values shape: " + str(np.shape(self.cache[-1].output_gate)))
        # calculate output from new state and output gate
        cache.output_values = Layer.activation_tanh(cache.state) * cache.output_gate_results
        ##print("ff output shape: " + str(np.shape(self.cache[-1].output_values)) + "\nff " + str(self.cache[-1].output_values))
        return cache.output_values

    def learn_recursive(self, cache, target_outputs, loss_total=0):
        debug("learn_rec(" + str(target_outputs) + ")")
        if len(target_outputs) == 0:
            return loss_total
        # calculate loss and cumulative loss but keep loss for t+1 (last loss)
        target = target_outputs[-1]
        loss_total += self.loss_function(cache.output_values[:], target)
        loss_output = 2 * (cache.output_values - target)
        loss_output += cache.successor.loss_output
        last_loss_state = cache.successor.loss_state

        delta_state = cache.output_gate_results * loss_output + last_loss_state

        delta_output_gate = self.output_gate_layer.activation_deriv(
            cache.output_gate_results) * cache.state * loss_output

        delta_input_gate = self.input_gate_layer.activation_deriv(
            cache.input_gate_results) * cache.update_values_layer_results * delta_state

        delta_update_values_layer = self.update_values_layer.activation_deriv(
            cache.update_values_layer_results) * cache.input_gate_results * delta_state

        delta_forget_gate = self.forget_gate_layer.activation_deriv(
            cache.forget_gate_results) * cache.predecessor.state * delta_state

        #print("delta_FG: " + str(np.shape(delta_forget_gate)))

        concat_in = cache.concatenated_input

        """print("input weights shape: " + str(np.shape(self.pending_updates.input_gate_weights)))
        print("input delta shape: " + str(np.shape(delta_input_gate)))
        print("concat in shape: " + str(np.shape(concat_in)))"""
        self.pending_updates.input_gate_weights += \
            np.outer(delta_input_gate, concat_in)
        self.pending_updates.forget_gate_weights += \
            np.outer(delta_forget_gate, concat_in)
        self.pending_updates.output_gate_weights += \
            np.outer(delta_output_gate, concat_in)
        self.pending_updates.update_values_layer_weights += \
            np.outer(delta_update_values_layer, concat_in)

        """print("input delta shape: " + str(np.shape(delta_input_gate)))
        print("input delta: " + str(np.reshape(delta_input_gate, (3,))))
        print("input_gate_biases pending: " + str(self.pending_updates.input_gate_biases))"""
        self.pending_updates.input_gate_biases += np.ravel(delta_input_gate)
        self.pending_updates.forget_gate_biases += np.ravel(delta_forget_gate)
        self.pending_updates.output_gate_biases += np.ravel(delta_output_gate)
        self.pending_updates.update_values_layer_biases += np.ravel(delta_update_values_layer)

        #TODO: compute bottom diff?
        delta_concatinated_input = np.zeros_like(concat_in) \
            + np.dot(self.input_gate_layer.weights.T, delta_input_gate)\
            + np.dot(self.forget_gate_layer.weights.T, delta_forget_gate)\
            + np.dot(self.output_gate_layer.weights.T, delta_output_gate)\
            + np.dot(self.update_values_layer.weights.T, delta_update_values_layer)

        cache.loss_state = delta_state * cache.forget_gate_results
        cache.loss_input = delta_concatinated_input[:self.in_size]
        cache.loss_output = delta_concatinated_input[self.in_size:]

        #print("calling learn_recursive with targets = " + str(target_outputs[:-1]))
        return self.learn_recursive(cache.predecessor, target_outputs[:-1], loss_total)

    def learn(self, target_outputs, learning_rate=0.001):
        debug("learn(" + str(target_outputs) + ")")
        loss = self.learn_recursive(self.last_cache.predecessor, target_outputs)
        self.apply_training(learning_rate)
        return loss

    def apply_training(self, learning_rate):
        p_updates = self.pending_updates
        lr = learning_rate
        #print("FG weights" + str(np.shape(self.forget_gate_layer.weights)))
        #print("FG weight updates" + str(np.shape(p_updates.forget_gate_weights)))
        self.forget_gate_layer.weights -= lr * p_updates.forget_gate_weights
        self.input_gate_layer.weights -= lr * p_updates.input_gate_weights
        self.update_values_layer.weights -= lr * p_updates.update_values_layer_weights
        self.output_gate_layer.weights -= lr * p_updates.output_gate_weights
        self.forget_gate_layer.biases -= lr * p_updates.forget_gate_biases
        self.input_gate_layer.biases -= lr * p_updates.input_gate_biases
        self.update_values_layer.biases -= lr * p_updates.update_values_layer_biases
        self.output_gate_layer.biases -= lr * p_updates.output_gate_biases
        p_updates.reset()

    @classmethod
    def get_output_error(cls, prediction, target):
        difference = np.zeros_like(prediction)
        tmp = 2 * np.sum(prediction - target)
        difference = tmp
        return difference

    @classmethod
    def euclidean_loss_function(cls, predicted, target):
        return (predicted - target) ** 2


class LSTMLayerPendingUpdates(object):
    def __init__(self, in_size, out_size):
        self.update_values_layer_weights = np.zeros((out_size, in_size + out_size))
        self.input_gate_weights = np.zeros((out_size, in_size + out_size))
        self.forget_gate_weights = np.zeros((out_size, in_size + out_size))
        self.output_gate_weights = np.zeros((out_size, in_size + out_size))
        self.update_values_layer_biases = np.zeros(out_size)
        self.input_gate_biases = np.zeros(out_size)
        self.forget_gate_biases = np.zeros(out_size)
        self.output_gate_biases = np.zeros(out_size)
        self.in_size = in_size
        self.out_size = out_size

    def reset(self):
        self.__init__(self.in_size, self.out_size)
