import numpy
import KTimage as KT

# directory for observing the results
path_obs = "res/"


class MLP:

    def __init__(self, size_in, size_out, size_hid):
        self.size_in = size_in
        self.size_hid = size_hid
        self.size_out = size_out

        self.weights[0] = numpy.random.uniform(-1.0, 1.0, (self.size_hid, self.size_in))
        self.weights[1] = numpy.random.uniform(-1.0, 1.0, (self.size_out, self.size_hid))

        self.activation_functions = [self.activation_function_linear,
                                     self.activation_function_linear,
                                     self.activation_function_linear]

    def feedforward(self, data, layer_id, results=[]):
        if layer_id < len(self.weights):
            results.append(data)
            return self.feedforward(self.activation_functions[layer_id](self.weights[layer_id], data), layer_id + 1, results)
        else:
            results.append(data)
            return results

    def backpropagate(self, results):
        # TODO: Calculate errors (cost)
        # TODO: Correct weights based on errors
        # TODO: backpropagate errors

        if len(results) != 0:

            self.backpropagate(results[:-1])

    def visualize(self, data):
        KT.exporttiles(data, self.size_in, 1, path_obs + "obs_S_0.pgm")
        KT.exporttiles(self.feedforward(data), self.size_out, 1, path_obs + "obs_S_1.pgm")
        KT.exporttiles(self.weights, self.size_in, 1, path_obs + "obs_W_1_0.pgm", self.size_out, 1)

    @staticmethod
    def activation_function_sigmoid(cls, weights, data):
        # TODO
        pass

    @staticmethod
    def activation_function_linear(cls, weights, data):
        return numpy.dot(weights, data)


data_in = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out = numpy.array([[0], [1], [1], [0]])

# slp = SLP(numpy.shape(data_in)[1], 1)
# slp.visualize(data_in[2])
