import numpy as np
from ..neural_network.NeuralLayer import \
    NeuralLayer as Layer, \
    BiasedNeuralLayer as BiasedLayer
from LSTMLayerCacheList import LSTMLayerCache


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
        self.caches = []
        self.pending_updates = LSTMLayerPendingUpdates(out_size, in_size + out_size)
        self.size = out_size
        self.in_size = in_size
        self.learning_rate = 0.001
        self.loss_function = LSTMLayer.euclidean_loss_function

        self.first_cache = LSTMLayerCache()
        self.last_cache = LSTMLayerCache()
        self.first_cache.state = np.zeros((self.size, 1))

    def feed(self, input_data, caching_depth=1):
        assert caching_depth > 0, "caching_depth must be at least 1 (for recursive input)!"
        ##print("ff input shape: " + str(np.shape(input_data)) + " - last output shape: " + str(np.shape(self.cache[-1].output_values)))
        data = np.concatenate([input_data, self.caches[-1].output_values])
        if caching_depth <= len(self.caches):
            self.caches.pop(0)
        self.caches.append(LSTMLayerCache())
        cache = self.caches[-1]
        if len(self.caches) == 1:
            cache.predecessor = self.first_cache
        else:
            self.caches[-2].successor = cache
            cache.predecessor = self.caches[-2]

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
            cache.input_gate,
            cache.update_values_layer_results)
        # apply forget layer and apply state update values
        ##print("ff forget gate: " + str(self.cache[-1].forget_gate))
        cache.state = cache.predecessor.state * self.caches[-1].forget_gate\
                      + update_values
        ##print("ff forgotten state shape: " + str(np.shape(self.cache[-1].state)))
        ##print("ff updated state shape: " + str(np.shape(self.cache[-1].state)))
        ##print("ff output values shape: " + str(np.shape(self.cache[-1].output_gate)))
        # calculate output from new state and output gate
        cache.output_values = Layer.activation_tanh(cache.state) * cache.output_gate
        ##print("ff output shape: " + str(np.shape(self.cache[-1].output_values)) + "\nff " + str(self.cache[-1].output_values))
        return cache.output_values

    def learn_recursive(self, cache, target_outputs, loss_total=0):
        if cache is None:
            return loss_total
        # calculate loss and cumulative loss but keep loss for t+1 (last loss)
        loss_total += self.loss_function(cache.output_values, target_outputs.pop())
        loss_output = 2 * (cache.output_values - target_outputs.pop())
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

        concat_in = cache.concatinated_input

        self.pending_updates.input_gate_weights += \
            np.outer(delta_input_gate, concat_in)
        self.pending_updates.forget_gate_weights += \
            np.outer(delta_forget_gate, concat_in)
        self.pending_updates.output_gate_weights += \
            np.outer(delta_output_gate, concat_in)
        self.pending_updates.update_values_layer_weights += \
            np.outer(delta_update_values_layer, concat_in)

        self.pending_updates.input_gate_biases += delta_input_gate
        self.pending_updates.forget_gate_biases += delta_forget_gate
        self.pending_updates.output_gate_biases += delta_output_gate
        self.pending_updates.update_values_layer_biases += delta_update_values_layer

        #TODO: compute bottom diff?
        delta_concatinated_input = np.zeros_like(concat_in) \
            + np.dot(self.input_gate_layer.weights.T, delta_input_gate)\
            + np.dot(self.forget_gate_layer.weights.T, delta_forget_gate)\
            + np.dot(self.output_gate_layer.weights.T, delta_output_gate)\
            + np.dot(self.update_values_layer.weights.T, delta_update_values_layer)

        cache.loss_state = delta_state * cache.forget_gate_results
        cache.loss_input = delta_concatinated_input[:self.in_size]
        cache.loss_output = delta_concatinated_input[self.in_size:]

        return self.learn_recursive(cache.predecessor, target_outputs, loss_total)

    def learn(self, target_outputs):
        self.caches[-1].successor = self.last_cache
        loss = self.learn_recursive(self.caches[-1], target_outputs)
        self.apply_training()
        return loss

    """def depr_learn(self, target_output):
        self.naive_learn(target_output)"""

    """def learn(self, target_output):
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
            #self.deltas.input_gate_weights += np.outer(delta_input_gate_act, cache.input_concat)
            self.deltas.input_gate_weights += np.outer(
                self.input_gate_layer.activation_deriv(cache.input_gate)
                    * cache.update_values * (cache.output_gate * output_error + state_error),
                cache.input_concat
            )

            # naive backprop
            delta_output_gate = np.outer(
                self.output_gate_layer.activation_deriv(cache.output_gate) * cache.state * output_error,
                cache.input_concat)
            delta_state...
            delta_update_values = np.outer(
                self.update_values_layer.activation_deriv(cache.update_values) * cache.input_gate * delta_state,
                cache.input_concat)

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
        return incremental_error"""

    def apply_training(self):
        p_updates = self.pending_updates
        lr = self.learning_rate
        self.forget_gate_layer.weights += lr * p_updates.forget_gate_weights
        self.input_gate_layer.weights += lr * p_updates.input_gate_weights
        self.update_values_layer.weights += lr * p_updates.update_values_layer_weights
        self.output_gate_layer.weights += lr * p_updates.output_gate_weights
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