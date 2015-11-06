from . import PerceptronLayer


class OutputLayer(PerceptronLayer):
    def __init__(self, in_size, out_size, activation_fn, activation_fn_deriv):
        super(in_size, out_size,
              activation_fn=PerceptronLayer.activation_sigmoid,
              activation_fn_deriv=PerceptronLayer.activation_sigmoid_deriv)

    def get_delta(self, result, error):
        return error * self.activation_deriv(result)