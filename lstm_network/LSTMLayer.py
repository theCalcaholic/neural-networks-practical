import os
import numpy as np
from neural_network.NeuralLayer import \
    NeuralLayer as Layer, \
    BiasedNeuralLayer as BiasedLayer
from LSTMLayerCacheList import LSTMLayerCache
from utils import debug
import KTimage


class LSTMLayer(object):
    def __init__(self, in_size, out_size, memory_size):
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
        self.output_gate_layer = Layer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of state
                activation_fn=Layer.activation_sigmoid,
                activation_fn_deriv=Layer.activation_sigmoid_deriv)

        self.use_output_layer = False
        self.use_output_slicing = False

        self.output_layer = Layer(
            in_size=memory_size,
            out_size=out_size,
            activation_fn=Layer.activation_linear,
            activation_fn_deriv=Layer.activation_linear_deriv)
        self.pending_updates = LSTMLayerPendingUpdates(in_size, memory_size, out_size)
        self.size = memory_size
        self.in_size = in_size
        self.out_size = out_size
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
        concat_in = np.concatenate([input_data, self.last_cache.predecessor.output_values])
        if caching_depth <= len(self.caches):
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
        # Uncomment for output layer:
        if self.use_output_layer:
            cache.final_output_values = self.output_layer.feed(cache.output_values)
        elif self.use_output_slicing:
            cache.final_output_values = cache.output_values[:self.out_size]
        else:
            cache.final_output_values = cache.output_values
        return cache.final_output_values

    def learn_recursive(self, cache, target_outputs, loss_total=0, learning_rate=0.01):
        debug("learn_rec(" + str(target_outputs) + ")")
        if len(target_outputs) == 0:
            return loss_total
        # calculate loss and cumulative loss but keep loss for t+1 (last loss)
        target = target_outputs[-1]
        #print(str(np.shape(target)) + str(target[0]) +
        #      "\n" + str(np.shape(cache.final_output_values)) + str(cache.final_output_values[0]))
        loss_total += self.loss_function(cache.final_output_values, target)

        # Uncomment for output layer
        delta_final_output = cache.final_output_values - target
        if self.use_output_layer:
            loss_output = np.dot(
                    self.output_layer.weights.T,
                    delta_final_output)
        elif self.use_output_slicing:
            loss_output = np.zeros((self.size, 1))
            loss_output[:self.out_size] = 2 * delta_final_output
        else:
            loss_output = 2 * delta_final_output
        loss_output += cache.successor.loss_output
        debug("loss_output: " + str(np.shape(loss_output))
              + "suc_loss_output: " + str(np.shape(cache.successor.loss_output)))

        last_loss_state = cache.successor.loss_state

        #print(str(np.shape(cache.final_output_values)) + str(np.shape(delta_final_output)))
        #self.output_layer.learn(cache.output_values, -delta_final_output, learning_rate)

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
        if self.use_output_layer:
            self.pending_updates.output_layer_weights += \
                np.outer(delta_final_output, cache.output_values)

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

        return self.learn_recursive(cache.predecessor, target_outputs[:-1], loss_total)

    def learn(self, target_outputs, learning_rate=0.001):
        debug("learn(" + str(target_outputs) + ")")
        loss = self.learn_recursive(self.last_cache.predecessor, target_outputs, learning_rate)
        self.apply_training(learning_rate)
        return loss

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
        if self.use_output_layer:
            self.output_layer.weights -= lr * p_updates.output_layer_weights
        p_updates.reset()

        for matrix in [
                self.forget_gate_layer.weights,
                self.input_gate_layer.weights,
                self.update_values_layer.weights,
                self.output_gate_layer.weights,
                self.forget_gate_layer.biases,
                self.input_gate_layer.biases,
                self.update_values_layer.biases,
                self.output_gate_layer.biases,
                self.output_layer.weights]:
            np.clip(matrix, -5, 5, out=matrix)

    def save(self, directory):
        self.forget_gate_layer.save(os.path.join(directory, "forget_gate.npz"))
        self.input_gate_layer.save(os.path.join(directory, "input_gate.npz"))
        self.output_gate_layer.save(os.path.join(directory, "output_gate.npz"))
        self.update_values_layer.save(os.path.join(directory, "update_values_layer.npz"))
        self.output_layer.save(os.path.join(directory, "output_layer.npz"))

    def load(self, directory):
        self.forget_gate_layer.load(os.path.join(directory, "forget_gate.npz"))
        self.input_gate_layer.load(os.path.join(directory, "input_gate.npz"))
        self.output_gate_layer.load(os.path.join(directory, "output_gate.npz"))
        self.update_values_layer.load(os.path.join(directory, "update_values_layer.npz"))
        self.output_layer.load(os.path.join(directory, "output_layer.npz"))

    @classmethod
    def get_output_error(cls, prediction, target):
        difference = np.zeros_like(prediction)
        tmp = 2 * np.sum(prediction - target)
        difference = tmp
        return difference

    @classmethod
    def euclidean_loss_function(cls, predicted, target):
        return (predicted - target) ** 2

    def visualize(self):
        KTimage.exporttiles(self.input_gate_layer.weights, self.in_size + self.size, 1, "visualize/obs_InputG_1_0.pgm",
                            1, self.size)
        KTimage.exporttiles(self.forget_gate_layer.weights, self.in_size + self.size, 1, "visualize/obs_ForgetG_2_0.pgm",
                            1, self.size)
        KTimage.exporttiles(self.update_values_layer.weights, self.in_size + self.size, 1,
                            "visualize/obs_UpdateL_3_0.pgm", 1, self.size)
        KTimage.exporttiles(self.output_gate_layer.weights, self.in_size + self.size, 1, "visualize/obs_OutputG_4_0.pgm",
                            1, self.size)
        KTimage.exporttiles(self.output_layer.weights, self.size, 1, "visualize/obs_OutputL_5_0.pgm", 1, self.output_layer.size)


class LSTMLayerPendingUpdates(object):
    def __init__(self, in_size, mem_size, out_size):
        self.update_values_layer_weights = np.zeros((mem_size, in_size + mem_size))
        self.input_gate_weights = np.zeros((mem_size, in_size + mem_size))
        self.forget_gate_weights = np.zeros((mem_size, in_size + mem_size))
        self.output_gate_weights = np.zeros((mem_size, in_size + mem_size))
        self.update_values_layer_biases = np.zeros(mem_size)
        self.input_gate_biases = np.zeros(mem_size)
        self.forget_gate_biases = np.zeros(mem_size)
        self.output_gate_biases = np.zeros(mem_size)
        self.output_layer_weights = np.zeros((out_size, mem_size))
        self.in_size = in_size
        self.mem_size = mem_size
        self.out_size = out_size

    def reset(self):
        self.__init__(self.in_size, self.mem_size, self.out_size)

    @classmethod
    def activation_fn_forget_gate(cls, x):
        return Layer.activation_sigmoid(x)
