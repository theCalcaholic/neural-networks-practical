import numpy
from mlp.MLP import PerceptronLayer as Layer, MLP

data_in = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out_xor = numpy.array([[0], [1], [1], [0]])

data_out_and = numpy.array([[0], [0], [0], [1]])

data_out_or = numpy.array([[0], [1], [1], [1]])

data_out = data_out_and

mlp = MLP(in_size=2,
          out_size=1,
          hidden_sizes=[3],
          hidden_fns=[Layer.activation_sigmoid],
          hidden_fns_deriv=[Layer.activation_sigmoid_deriv])

mlp.set_learning_rate(0.01)

mlp.train(data_in, data_out)

while True:
    test_data_str = raw_input("Enter data to be processed by the MLp (comma separated vector of size " +
                          str(len(data_in[0])) +
                          ")\n:")
    if not len(test_data_str):
        break
    test_data = [int(x) for x in test_data_str.split(",")]
    print("Result: " + str(mlp.predict(test_data)))
