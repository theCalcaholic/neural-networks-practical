class LSTMLayerCache(object):
    """
    caches the inputs, results and losses of an lstm layer feedforward process.
    Supports double linking (predecessor, successor)
    """
    def __init__(self):
        """initialize cache"""
        # inputs
        self.input_values = None
        self.concatenated_input = None
        # results
        self.output_gate_results = None
        self.update_values_layer_results = None
        self.input_gate_results = None
        self.forget_gate_results = None
        # state & output
        self.state = None
        self.output_values = None
        # loss
        self.loss_output = 0
        self.loss_state = 0
        self.loss_input = 0
        # list implementation stuff
        self.predecessor = None
        self.successor = None
        self.is_first_cache = False
        self.is_last_cache = False

    def insert_before(self, cache):
        """insert cache before this one in cache chain"""
        cache.predecessor = self.predecessor
        cache.successor = self
        self.predecessor.successor = cache
        self.predecessor = cache

    def insert_after(self, cache):
        """insert cache after this one in cache chain"""
        cache.successor = self.successor
        cache.predecessor = self
        self.successor = cache

    def remove(self):
        """remove this cache from cache chain"""
        self.predecessor.successor = self.successor
        self.successor.predecessor = self.predecessor
