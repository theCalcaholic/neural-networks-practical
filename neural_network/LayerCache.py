
class LayerCache(object):
    def __init__(self):
        # inputs
        self.input_values = None
        # results
        self.output_values = None
        # loss
        self.loss = None
        # list implementation stuff
        self.predecessor = None
        self.successor = None
        self.is_first_cache = False
        self.is_last_cache = False

    def insert_before(self, cache):
        cache.predecessor = self.predecessor
        cache.successor = self
        self.predecessor.successor = cache
        self.predecessor = cache

    def insert_after(self, cache):
        cache.successor = self.successor
        cache.predecessor = self
        self.successor = cache

    def remove(self):
        self.predecessor.successor = self.successor
        self.successor.predecessor = self.predecessor