import numpy as np
from lstm_network.LSTMNetwork import LSTMNetwork
from lstm_network.utils import decode


input_data = "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
    + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"
memory_size = 6
sequence_length = 25  # 'length' of memory
learning_rate = 1e-1
data_set = set(input_data)

data_to_int = {dat: i for i, dat in enumerate(data_set)}
int_to_data = {i: dat for i, dat in enumerate(data_set)}

encoded_data = []
for t in xrange(len(input_data)):
    # encode character in 1-of-k representation
    input_int = data_to_int[input_data[t]]
    encoded_data.append(np.zeros((len(data_set), 1)))
    encoded_data[-1][input_int] = 1

lstm = LSTMNetwork()
lstm.int_to_data = int_to_data

lstm.populate(data_set, layer_sizes=[memory_size])

lstm.train(encoded_data, 5, iterations=300)


#print("result: " + str(lstm.feedforward(np.array([1]))))
#print("result: " + str(lstm.feedforward(np.array([1]))))

while True:
    test_data_str = raw_input("Enter data to be predicted by the LSTM Network (char of alphabet: " +
                          str(data_set) + "):\n")
    if not len(test_data_str):
        break
    num = int(raw_input("Enter amount of characters to predict."))
    inp = np.zeros((len(data_set), 1))
    inp[data_to_int[test_data_str]] = 1
    results = lstm.predict(inp, num)
    out = ""
    for i in range(num):
        out += decode(results[i], int_to_data)
    print("Result: " + out)
