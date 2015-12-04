from . import PerceptronLayer


class InputLayer(PerceptronLayer):
    def __init__(self,
                 in_size,
                 out_size):
        super(in_size, out_size,
              Woche2.PerceptronLayer.activation_linear,
              Woche2.PerceptronLayer.activation_linear_deriv)