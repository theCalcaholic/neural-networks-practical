from PerceptronLayer import PerceptronLayer


class OutputLayer(PerceptronLayer):
    def __init__(self, in_size, out_size):
        print("Creating output layer(in: " + str(in_size) + ", out: " + str(out_size) + ")...")
        self.layer_type = "Output Layer"
        super(OutputLayer, self).__init__(in_size, out_size,
                                          activation_fn=PerceptronLayer.activation_linear,
                                          activation_fn_deriv=PerceptronLayer.activation_linear_deriv)

