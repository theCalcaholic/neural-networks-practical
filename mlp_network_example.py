import numpy as np
from neural_network.util import Logger
from neural_network.NeuralLayer import NeuralLayer as Layer
from multi_layer_perceptron.MLPNetwork import MLPNetwork

Logger.DEBUG = False

data_in = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
data_out_xor = np.array([[0], [1], [1], [0]])

data_out_and = np.array([[0], [0], [0], [1]])

data_out_or = np.array([[0], [1], [1], [1]])

data_out = data_out_xor

Logger.log("data_shape: " + str(np.shape(data_in)))
Logger.log("data[0]_shape: " + str(np.shape(data_in[0])))

mlp = MLPNetwork()

mlp.populate(
        in_size=2,
        out_layer= {
            "size": 1,
            "fn": None,
            "fn_deriv": None
        },
        hidden_layers=[{
            "size": 3,
            "fn": Layer.activation_sigmoid,
            "fn_deriv": Layer.activation_sigmoid_deriv
        }]
)

mlp.configure({
    "learning_rate": 0.5,
    "verbose": True,
    "status_frequency": 100
})


mlp.train(
        inputs=data_in,
        targets=data_out,
        iterations=1000
)

while True:
    test_data_str = raw_input("Enter data to be predicted by the MLp (comma separated vector of size " +
                          str(len(data_in[0])) + "):\n")
    if not len(test_data_str):
        break
    test_data = np.array([[int(x)] for x in test_data_str.split(",")])
    print("Result: " + str(mlp.predict(test_data)))