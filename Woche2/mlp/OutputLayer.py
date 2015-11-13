from PerceptronLayer import PerceptronLayer
import numpy


class OutputLayer(PerceptronLayer):
    def __init__(self, in_size, out_size):
        print("Creating output layer(in: " + str(in_size) + ", out: " + str(out_size) + ")...")
        self.layer_type = "Output Layer"
        super(OutputLayer, self).__init__(in_size, out_size,
                                          activation_fn=PerceptronLayer.activation_linear,
                                          activation_fn_deriv=PerceptronLayer.activation_linear_deriv)

    """def get_delta(self, result, error):
        return error * self.activation_deriv(result)"""

    def get_delta(self, result, last_delta, last_weights):
        #last_delta = numpy.atleast_2d(last_delta)
        #print("Delta:", last_delta)
        last_weights = numpy.atleast_2d(last_weights)
        #print("Weights:", last_weights)
        #return numpy.outer(last_delta, last_weights)
        return numpy.dot(last_delta, last_weights) * self.activation_deriv(result)
