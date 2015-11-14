import numpy
from mlp.MLP import PerceptronLayer as Layer, MLP

data_in_xor = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out_xor = numpy.array([[0], [1], [1], [0]])

data_in_and = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out_and = numpy.array([[0], [0], [0], [1]])

data_in_or = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out_or = numpy.array([[0], [1], [1], [1]])

data_in = data_in_xor
data_out = data_out_xor

mlp = MLP(in_size=2,
          out_size=1,
          hidden_sizes=[3],
          hidden_fns=[Layer.activation_sigmoid],
          hidden_fns_deriv=[Layer.activation_sigmoid_deriv])

mlp.set_learning_rate(0.01)

mlp.train(data_in, data_out)
