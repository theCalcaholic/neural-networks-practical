from . import PerceptronLayer


class HiddenLayer(PerceptronLayer):
    def __init__(self,
                in_size,
                out_size,
                activation_fn=Woche2.mlp.PerceptronLayer.activation_sigmoid,
                activation_fn_deriv=Woche2.mlp.PerceptronLayer.activation_sigmoid_deriv):
        super(in_size, out_size, activation_fn, activation_fn_deriv)

    def get_delta(self, delta, last_weights, result):
        delta.dot(last_weights.T) * self.activation_deriv(result)