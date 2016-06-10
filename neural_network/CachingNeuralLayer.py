from neural_network import NeuralLayer, BiasedNeuralLayer
from neural_network import LayerCache
from neural_network.util import Logger
import numpy as np


class CachingNeuralLayer(NeuralLayer):
    """NeuralLayer that supports caching its inputs, results and losses"""
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        """initialize layer (compare class NeuralLayer)"""
        # initialize cache chain (self.first_cache and self.last_cache are dummy caches)
        self.first_cache = LayerCache()
        self.first_cache.is_first_cache = True
        self.last_cache = LayerCache()
        self.last_cache.is_last_cache = True
        self.first_cache.successor = self.last_cache
        self.last_cache.predecessor = self.first_cache
        self.caches = []
        # invoke super constructor
        super(CachingNeuralLayer, self).__init__(in_size, out_size, activation_fn, activation_fn_deriv)

    def feed(self, input_data, time_steps=1):
        """calculate result for input vector and cache intermediate results"""
        # if number of cache exceeds <tim_steps>, remove first cache
        if time_steps <= len(self.caches):
            self.first_cache.successor.remove()
            self.caches.pop(0)

        # create new cache
        cache = LayerCache()
        self.last_cache.insert_before(cache)
        self.caches.append(cache)
        # cache input vector
        cache.input_values = input_data
        # calculate and cache output vector
        cache.output_values = super(CachingNeuralLayer, self).feed(input_data)
        return cache.output_values

    def get_deltas(self, last_deltas, last_weights):
        return [super(CachingNeuralLayer, self).get_delta(cache.output_values, last_delta, last_weights)
                for cache, last_delta in zip(self.caches, last_deltas)]

    def learn(self, result, delta, learning_rate):
        #Logger.log("(expected delta: " + str(np.shape(self.caches[0].output_values)) + ")")
        #Logger.debug("delta: " + str(delta) + "\nresult:" + str(result))
        super(CachingNeuralLayer, self).learn(result, delta, learning_rate)

class CachingBiasedNeuralLayer(BiasedNeuralLayer, CachingNeuralLayer):
    """BiasedNeuralLayer that supports caching"""
    pass
