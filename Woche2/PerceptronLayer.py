import numpy


class PerceptronLayer:
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        self.weights = numpy.random.uniform(-1.0, 1.0, out_size, in_size)
        self.activation = activation_fn
        self.reverse_activation = activation_fn_deriv

    def