import os
import numpy as np
from neural_network import \
    NeuralLayer as Layer, \
    BiasedNeuralLayer as BiasedLayer
from LSTMLayerCache import LSTMLayerCache
from neural_network.util import Logger


class LSTMLayer(object):
    """
    represents a complete lstm layer (resp. lstm block) while offering a similar interface as the NeuralLayer class
    """
    def __init__(self, in_size, memory_size):
        """
        initialize lstm layer
        @in_size size of expected input vectors
        @memory_size size of memory vector (state)
        """
        concat_size = in_size + memory_size
        # create forget layer using sigmoid activation function
        self.forget_gate_layer = BiasedLayer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of state
                activation_fn=Layer.activation_sigmoid,
                activation_fn_deriv=Layer.activation_sigmoid_deriv)
        # initialize forget gate layer biases to 1
        self.forget_gate_layer.biases = np.ones(memory_size)
        # create input gate layer using sigmoid activation function
        self.input_gate_layer = BiasedLayer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of update_values_layer (transposed state)
                activation_fn=Layer.activation_sigmoid,
                activation_fn_deriv=Layer.activation_sigmoid_deriv)
        # create update values layer using tanh activation function
        self.update_values_layer = BiasedLayer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of state
                activation_fn=Layer.activation_tanh,
                activation_fn_deriv=Layer.activation_tanh_deriv)
        # create output gate layer using sigmoid activation function
        self.output_gate_layer = BiasedLayer(
                in_size=concat_size,  # size of input/last output
                out_size=memory_size,  # size of state
                activation_fn=Layer.activation_sigmoid,
                activation_fn_deriv=Layer.activation_sigmoid_deriv)
        # create container object for pending updates
        self.pending_updates = LSTMLayerPendingUpdates(in_size, memory_size)
        self.size = memory_size
        self.in_size = in_size

        #initialize cache chain (self.first_cache and self.last_cache are dummy caches)
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
        """returns a copy of the lstm layer which shares the same weights and biases"""
        conv_layer = LSTMLayer(
            in_size=reference_layer.in_size,
            memory_size=reference_layer.size
        )
        # set the second layer's gates and update values layer to this one's
        conv_layer.input_gate_layer = reference_layer.input_gate_layer
        conv_layer.forget_gate_layer = reference_layer.forget_gate_layer
        conv_layer.output_gate_layer = reference_layer.output_gate_layer
        conv_layer.update_values_layer = reference_layer.update_values_layer

        return conv_layer

    def feed(self, input_data, time_steps=1):
        """
        calculate output vector for given input vector
        @time_steps number of feedforward results to be cached for use in backpropagation through time
        """
        assert time_steps > 0, "time_steps must be at least 1 (for recursive input)!"
        Logger.debug("feed(" + str(input_data) + ")")

        # concatenate input_vector with recurrent input (last output) vector
        concat_in = np.concatenate([input_data, self.last_cache.predecessor.output_values])
        # delete first cache if maximum number of time_steps are already cached
        if time_steps <= len(self.caches):
            self.first_cache.successor.remove()
            self.caches.pop(0)

        # create new cache at end of cache list
        self.last_cache.insert_before(LSTMLayerCache())
        cache = self.last_cache.predecessor
        self.caches.append(cache)

        # cache input and concatenated input values
        cache.input_values = input_data
        cache.concatenated_input = concat_in

        # calculate and cache gate/update_values results
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

        # return calculated output vector
        return cache.output_values

    def learn_recursive(self, cache, deltas):
        """
        learn timesteps recursively
        @cache cache corresponding to current time step
        @deltas deltas from last (actually next) layer
        """
        # terminate if there are no target values or caches left
        if len(deltas) == 0 or cache.is_first_cache:
            return
        # get delta for current time step
        delta = deltas[-1]

        # calculate cumulative loss derived with respect to output (CEC - Constant Error Carousel)
        loss_output = delta + cache.successor.loss_output

        # retrieve loss from last time step (t+1) derived wrt state
        last_loss_state = cache.successor.loss_state

        ### calculate deltas
        delta_state = cache.output_gate_results * loss_output + last_loss_state

        delta_output_gate = self.output_gate_layer.activation_deriv(
                cache.output_gate_results) * cache.state * loss_output

        delta_input_gate = self.input_gate_layer.activation_deriv(
                cache.input_gate_results) * cache.update_values_layer_results * delta_state

        delta_update_values_layer = self.update_values_layer.activation_deriv(
                cache.update_values_layer_results) * cache.input_gate_results * delta_state

        delta_forget_gate = self.forget_gate_layer.activation_deriv(
                cache.forget_gate_results) * cache.predecessor.state * delta_state
        ###

        # retrieve concatenated input from cache
        concat_in = cache.concatenated_input

        # add weight adjustments to pending updates object
        self.pending_updates.input_gate_weights += \
            np.outer(delta_input_gate, concat_in)
        self.pending_updates.forget_gate_weights += \
            np.outer(delta_forget_gate, concat_in)
        self.pending_updates.output_gate_weights += \
            np.outer(delta_output_gate, concat_in)
        self.pending_updates.update_values_layer_weights += \
            np.outer(delta_update_values_layer, concat_in)

        # add bias adjustments to pending updates object
        self.pending_updates.input_gate_biases += np.ravel(delta_input_gate)
        self.pending_updates.forget_gate_biases += np.ravel(delta_forget_gate)
        self.pending_updates.output_gate_biases += np.ravel(delta_output_gate)
        self.pending_updates.update_values_layer_biases += np.ravel(delta_update_values_layer)

        # calculate loss with respect to concatenated input
        delta_concatinated_input = np.zeros_like(concat_in) + \
                                   np.dot(self.input_gate_layer.weights.T, delta_input_gate) + \
                                   np.dot(self.forget_gate_layer.weights.T, delta_forget_gate) + \
                                   np.dot(self.output_gate_layer.weights.T, delta_output_gate) + \
                                   np.dot(self.update_values_layer.weights.T, delta_update_values_layer)

        # save loss for Constant Error Carousel
        cache.loss_state = delta_state * cache.forget_gate_results
        cache.loss_input = delta_concatinated_input[:self.in_size]
        cache.loss_output = delta_concatinated_input[self.in_size:]

        # call itself recursively for next time step (t-1)
        return self.learn_recursive(cache.predecessor, deltas[:-1])

    def learn(self, deltas, learning_rate=0.001):
        """
        apply learning algorithm by using deltas from next layer
        @deltas deltas from last (actually next) layer
        """
        Logger.debug("learn(" + str(deltas) + ")")
        # learn recursively over all caches (corresponds to time steps), starting with last cache
        self.learn_recursive(self.last_cache.predecessor, deltas)
        # apply pending weight and bias updates
        self.apply_training(learning_rate)
        # calculate and return deltas for this layer from losses
        deltas = [cache.loss_input for cache in self.caches]
        return deltas

    def apply_training(self, learning_rate):
        """applies the calculated weight and bias updates and resets pending_updates object"""
        p_updates = self.pending_updates
        lr = learning_rate
        # subtract updates multiplied with learning rate from weight matrices/bias vectors
        self.forget_gate_layer.weights -= lr * p_updates.forget_gate_weights
        self.input_gate_layer.weights -= lr * p_updates.input_gate_weights
        self.update_values_layer.weights -= lr * p_updates.update_values_layer_weights
        self.output_gate_layer.weights -= lr * p_updates.output_gate_weights
        self.forget_gate_layer.biases -= lr * p_updates.forget_gate_biases
        self.input_gate_layer.biases -= lr * p_updates.input_gate_biases
        self.update_values_layer.biases -= lr * p_updates.update_values_layer_biases
        self.output_gate_layer.biases -= lr * p_updates.output_gate_biases
        # reset pending updates
        p_updates.reset()

        # clip matrices to prevent exploding gradient
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
        """save weights and biases to directory"""
        self.forget_gate_layer.save(os.path.join(directory, "forget_gate.npz"))
        self.input_gate_layer.save(os.path.join(directory, "input_gate.npz"))
        self.output_gate_layer.save(os.path.join(directory, "output_gate.npz"))
        self.update_values_layer.save(os.path.join(directory, "update_values_layer.npz"))

    def load(self, directory):
        """load weights and biases from directory"""
        self.forget_gate_layer.load(os.path.join(directory, "forget_gate.npz"))
        self.input_gate_layer.load(os.path.join(directory, "input_gate.npz"))
        self.output_gate_layer.load(os.path.join(directory, "output_gate.npz"))
        self.update_values_layer.load(os.path.join(directory, "update_values_layer.npz"))

    def visualize(self, path, layer_id):
        """generate visualization of weights and biases"""
        self.input_gate_layer.visualize(os.path.join(path, "LSTM" + str(layer_id), "obs_" + "InputG_1_0.pgm"))
        self.forget_gate_layer.visualize(os.path.join(path, "LSTM" + str(layer_id), "obs_" + "ForgetG_2_0.pgm"))
        self.output_gate_layer.visualize(os.path.join(path, "LSTM" + str(layer_id), "obs_" + "OutputG_3_0.pgm"))
        self.update_values_layer.visualize(os.path.join(path, "LSTM" + str(layer_id), "obs_" + "UpdateL_4_0.pgm"))

    def clear_cache(self):
        """clear all caches (i.e. state , error carousel, layer results)"""
        self.caches = []
        self.first_cache.successor = self.last_cache
        self.last_cache.predecessor = self.first_cache


class LSTMLayerPendingUpdates(object):
    """Serves as a container for pending updates for weight matrices and bias vectors"""
    def __init__(self, in_size, mem_size):
        """initialize pending updates object"""
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
        """reinitialize pending updates object"""
        self.__init__(self.in_size, self.mem_size)

