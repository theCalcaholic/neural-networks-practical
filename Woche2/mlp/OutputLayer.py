from . import PerceptronLayer


class OutputLayer(PerceptronLayer):
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        super(in_size, out_size, activation_fn, activation_fn_deriv)