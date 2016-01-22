import numpy as np
from lstm_network.LSTMNetwork import LSTMNetwork
from lstm_network.utils import decode, encode
import random


input_data = [line.rstrip('\n').replace(' ', '-') for line in open("lstm_training_data/schimpfworte.txt")]

sequence_length = 50  # 'length' of memory
learning_rate = 0.1
memory_size = 100
data_set = set([ord(x) for x in ' '.join(input_data)])
data_set = data_set.union(set([(x + 256) for x in range(abs(memory_size - len(set(data_set))))]))

_data_to_int = {dat: i for i, dat in enumerate(data_set)}
_int_to_data = {i: dat for i, dat in enumerate(data_set)}

def data_to_int(x):
    return _data_to_int[x]

def int_to_data(x):
    return "#" if _int_to_data[x] > 255 else chr(_int_to_data[x])

ids = range(len(input_data) - 1)

random_samples = []
for i in range(50):
    rand_num = random.randint(0, len(ids))
    rand_id = ids.pop(rand_num)
    random_samples.append(input_data[rand_id])
random_samples = [ord(x) for x in ' '.join(random_samples)]
#print(random_samples)
encoded_data = []
for t in xrange(len(random_samples)):
    # encode character in 1-of-k representation
    encoded_data.append(encode(random_samples[t], data_to_int, len(data_set)))

#print(str(encoded_data[-1]))
lstm = LSTMNetwork()

#print(str(len(data_set)) + "\nmemsize " + str(memory_size) + "\nmaxInt")

lstm.populate(data_set, layer_sizes=[memory_size])
#lstm.load("lstm_schimpfworte_save")

lstm._int_to_data = _int_to_data
lstm._data_to_int = _data_to_int
lstm.train(encoded_data, 20, iterations=20000, learning_rate=learning_rate, save_dir="lstm_schimpfworte_save")


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
