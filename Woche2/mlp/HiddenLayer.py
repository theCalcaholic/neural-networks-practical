from PerceptronLayer import PerceptronLayer
import numpy


class HiddenLayer(PerceptronLayer):
    def __init__(self, in_size, out_size, activation_fn=PerceptronLayer.activation_sigmoid, activation_fn_deriv=PerceptronLayer.activation_sigmoid_deriv):
        self.layer_type = "Hidden Layer"
        print("Creating hidden layer(in: " + str(in_size) + ", out: " + str(out_size) + ")...")
        super(HiddenLayer, self).__init__(in_size, out_size,
                                          activation_fn,
                                          activation_fn_deriv)

    def get_delta(self, result, last_delta, last_weights):
        #last_delta = numpy.atleast_2d(last_delta)
        #print("Delta:", last_delta)
        last_weights = numpy.atleast_2d(last_weights)
        #print("Weights:", last_weights)
        #return numpy.outer(last_delta, last_weights)
        return numpy.dot(last_delta, last_weights) * self.activation_deriv(result)