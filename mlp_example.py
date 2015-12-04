import numpy

from neural_network.NeuralLayer import PerceptronLayer as Layer
from multi_layer_perceptron.MLP import MLP

data_in = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data_out_xor = numpy.array([[0], [1], [1], [0]])

data_out_and = numpy.array([[0], [0], [0], [1]])

data_out_or = numpy.array([[0], [1], [1], [1]])

data_out = data_out_xor

mlp = MLP(in_size=2,
          layers=[
              {
                  "size": 3,
                  "fn": Layer.activation_sigmoid,
                  "fn_deriv": Layer.activation_sigmoid_deriv
              },
              {
                  "size": 1,
                  "fn": None,
                  "fn_deriv": None
              }
          ])

mlp.configure({
    "learning_rate": 0.02,
    "target_precision": 0.0001,
    "max_iterations": 100000
})

mlp.train(data_in, data_out)

while True:
    test_data_str = raw_input("Enter data to be predicted by the MLp (comma separated vector of size " +
                          str(len(data_in[0])) + "):\n")
    if not len(test_data_str):
        break
    test_data = [int(x) for x in test_data_str.split(",")]
    print("Result: " + str(mlp.predict(test_data)))
