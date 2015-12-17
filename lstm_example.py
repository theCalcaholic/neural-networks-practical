import numpy as np
from lstm_network.LSTMNetwork import LSTMNetwork

lstm = LSTMNetwork()

lstm.populate(1, [2, 1])

print("result: " + str(lstm.feedforward(np.array([1]))))
print("result: " + str(lstm.feedforward(np.array([1]))))
