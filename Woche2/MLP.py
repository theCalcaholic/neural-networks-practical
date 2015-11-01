import numpy
import KTimage as KT

# directory for observing the results
path_obs = "res/"


class MLP:

    def __init__(self, size_in, size_out, size_hid, activation_fns, activation_fn_derivs):
        #if layers < 2:
        #    raise Exception("At least two layers required!")
        #self.layers = layers
        self.act_fns = activation_fns
        self.act_fn_derivs = activation_fn_derivs
        self.size_in = size_in
        self.size_out = size_out
        self.size_hid = size_hid


        self.weights = []
        self.weights[0] = numpy.random.uniform(-1.0, 1.0, (self.size_hid, self.size_in))
        self.weights[1] = numpy.random.uniform(-1.0, 1.0, (self.size_out, self.size_hid))

        self.activation_functions = [self.activation_function_linear,
                                     self.activation_function_linear,
                                     self.activation_function_linear]

    def feedforward(self, data, layer_id, results=[]):
        if layer_id < len(self.weights):
            results.append([data])
            return self.feedforward(
                self.linear_activation(self.weights[layer_id] * data), layer_id + 1, results)
        else:
            results.append([data])
            return results[1:]

    def backpropagate(self, results, training_data):
        # TODO: Calculate errors (cost)
        # TODO: Correct weights based on errors
        # TODO: backpropagate errors
        learning_rate = 0.1

        #corrections = numpy.array.zeros(self.weights)

        delta2 = training_data - results[2]

        delta1 = numpy.dot(delta2, self.weights[1]) #TODO: Ableitung v. Akt.-Funktion

        self.weights[1] = numpy.subtract(
            self.weights[1],
            numpy.dot(learning_rate, numpy.dot(delta2, results[1])))
        self.weights[0] = numpy.subtract(
            self.weights[0],
            numpy.dot(learning_rate, numpy.dot(delta1, results[0])))

        print "Error: ", (training_data[0] - results[2])

    def train(self, input, output):
        number_of_iterations = 1000
        for i in range(number_of_iterations):
            for j in input.iteritems():
                self.backpropagate(self.feedforward(input[i], 0), output[i])

    def visualize(self, data):
        KT.exporttiles(data, self.size_in, 1, path_obs + "obs_S_0.pgm")
        KT.exporttiles(self.feedforward(data), self.size_out, 1, path_obs + "obs_S_1.pgm")
        KT.exporttiles(self.weights, self.size_in, 1, path_obs + "obs_W_1_0.pgm", self.size_out, 1)

    @staticmethod
    def activation_function_sigmoid(weights, data):
        # TODO
        pass

    @staticmethod
    def activation_function_linear(weights, data):
        return numpy.dot(weights, data)


data_in = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out = numpy.array([[0], [1], [1], [0]])

mlp = MLP(2, 1, 3, None, None)

MLP.train(data_in, data_out)

# slp = SLP(numpy.shape(data_in)[1], 1)
# slp.visualize(data_in[2])
