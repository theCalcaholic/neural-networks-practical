import Theano as theano
import numpy


class HiddenLayer:
    def _init__(self, input, n_in, n_out, W=None, activation=(lambda x, w: x*w)):
        self.input = input
        if W is None:
            W_values = numpy.random.uniform(-1.0, 1.0, n_out, n_in)
            W = theano.shared(value=W_values)
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano)
            b = theano.shared(value=b_values, name='b')
            self.W = W
            self.b = b
