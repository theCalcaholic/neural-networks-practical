from neural_network import NeuralLayer, BiasedNeuralLayer
from neural_network import LayerCache


class CachingNeuralLayer(NeuralLayer):
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        self.first_cache = LayerCache()
        self.first_cache.is_first_cache = True
        self.last_cache = LayerCache()
        self.last_cache.is_last_cache = True
        self.first_cache.successor = self.last_cache
        self.last_cache.predecessor = self.first_cache
        self.caches = []
        super(CachingNeuralLayer, self).__init__(in_size, out_size, activation_fn, activation_fn_deriv)

    def feed(self, input_data, time_steps=1):
        if time_steps <= len(self.caches):
            self.first_cache.successor.remove()
            self.caches.pop(0)

        cache = LayerCache()
        self.last_cache.insert_before(cache)
        self.caches.append(cache)
        cache.input_values = input_data
        cache.output_values = super(CachingNeuralLayer, self).feed(input_data)
        return cache.output_values


class CachingBiasedNeuralLayer(BiasedNeuralLayer, CachingNeuralLayer):
    pass
