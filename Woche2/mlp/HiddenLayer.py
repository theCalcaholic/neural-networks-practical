from PerceptronLayer import PerceptronLayer


class HiddenLayer(PerceptronLayer):
    def __init__(self, in_size, out_size, activation_fn=PerceptronLayer.activation_sigmoid, activation_fn_deriv=PerceptronLayer.activation_sigmoid_deriv):
        super(HiddenLayer, self).__init__(in_size, out_size,
                                          activation_fn,
                                          activation_fn_deriv)
