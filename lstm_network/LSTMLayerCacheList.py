import numpy as np


class LSTMLayerCacheList(object):
    def __init__(self):
        self.first = None
        self.last = None
        self.length = 0

    def _get_actual_index(self, x):
        if x is None:
            x = -1
        if x < 0:
            x += self.length
        return x

    def _get_xth_item(self, x):
        item = None
        if x <= self.length / 2:
            i = 0
            item = self.first
            while i < x:
                try:
                    item = item.successor
                except AttributeError:
                    raise IndexError
        elif x > self.length / 2:
            i = 0
            x = self.length - x
            item = self.last
            while i < x:
                try:
                    item = item.predecessor
                except AttributeError:
                    raise IndexError
        return item

    def __setitem__(self, key, value):
        if key == self.length:
            self.last.successor = value
            self.last = value
        elif key > self.length:
            raise IndexError
        else:
            pre = self._get_xth_item(key - 1)
            pre.successor.predecessor = value
            pre.successor = value

    def __iadd__(self, other):
        self.add(other)

    def __getitem__(self, x):
        return self._get_xth_item(x)

    def __delitem__(self, key):
        self.delete(key)

    def __iter__(self):
        return self.Iterator(self.first)

    def __reversed__(self):
        return self.Iterator(self.last, reverse=True)

    def add(self, cache):
        if not isinstance(cache, LSTMLayerCache):
            cache = LSTMLayerCache(cache)

        if self.first is None:
            self.first = cache
            self.last = cache
        else:
            self.last.successor = cache
            cache.predecessor = self.last
            self.last = cache
        self.length += 1

    def append(self, cache):
        self.add(cache)

    def delete(self, x=None):
        x = self._get_actual_index(x)
        del_cache = self._get_xth_item(x)

        del_cache.predecessor.successor = del_cache.successor
        del_cache.successor.predecessor = del_cache.predecessor
        self.length -= 1
        return del_cache

    def pop(self, x=None):
        self.delete(x)

    def deque(self, x=None):
        self.delete(x)

    def remove(self, x=None):
        self.delete(x)

    class Iterator(object):
        def __init__(self, list_start, list_end, reverse=False):
            self.list_start = list_start
            self.list_end = list_end
            self.reverse = reverse
            if reverse:
                self.cur = self.list_end
                self.__next__ = self.next_rev
            else:
                self.cur = self.list_start
                self.__next__ = self.next_ord

        def __iter__(self):
            return self

        def __reversed__(self):
            return self.__class__(self.list_start, self.list_end, reverse=(not self.reverse))

        def next_ord(self):
            if self.cur.successor is None:
                raise StopIteration
            else:
                return self.cur.successor

        def next_rev(self):
            if self.cur.predecessor is None:
                raise StopIteration
            else:
                return self.cur.predecessor


class LSTMLayerCache(object):
    def __init__(self):
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

