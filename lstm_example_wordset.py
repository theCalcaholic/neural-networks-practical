import numpy as np
from lstm_network.LSTMNetwork import LSTMNetwork
from lstm_network.utils import decode, encode
from DataStore import DataStore
import random

"""input_text = "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"""""
input_text = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, ipsum sed diam"
data_store = DataStore()
data_store.set_input_text(input_text)
#data_store.load_file("lstm_training_data/vanishing_vocables_de.txt")
#data_store.load_file("lstm_training_data/loremipsum.txt")
data_store.configure({
    "memory_size": 100,
    "sequence_length": 15,
    "extend_alphabet": False
})

samples = data_store.get_samples()
#print(data_store.decode_data_list(data_store.data_to_int.items()))
#print(str(encoded_data[-1]))
lstm = LSTMNetwork()

#print(str(len(data_set)) + "\nmemsize " + str(memory_size) + "\nmaxInt")

lstm.use_output_layer = False
lstm.use_output_slicing = False
lstm.populate(data_store.data_set,
              layer_sizes=[data_store.config["memory_size"]])
#lstm.load("lstm_wordset_save")
lstm.decode = data_store.decode_data_list
lstm.encode = data_store.encode_string
lstm.train(
        samples,
        seq_length=data_store.config["sequence_length"],
        iterations=20000,
        learning_rate=0.01,
        save_dir="lstm_loremipsum_save")

configs = {
    "lorem_ipsum_small": {
        "memory_size": 200,
        "sequence_length": 10,
        "learning_rate": 0.1
    }
}