from lstm_network.LSTMNetwork import LSTMNetwork
from DataStore import DataStore
import os
import numpy as np

config_list = {
    "lorem_ipsum_small": {
        "memory_size": 300,
        "sequence_length": 30,
        "learning_rate": 0.1,
        "iterations": 10000,
        "target_loss": 0.05,
        "use_output_layer": True
    }
}

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

config = config_list["lorem_ipsum_small"]


data_store.configure({
    "memory_size": config["memory_size"],
    "sequence_length": config["sequence_length"],
    "extend_alphabet": False
})

samples = data_store.get_samples()
#print(data_store.decode_data_list(data_store.data_to_int.items()))
#print(str(encoded_data[-1]))
lstm = LSTMNetwork()


lstm.use_output_layer = config["use_output_layer"]
lstm.populate(
    in_size=np.shape(samples[0])[0],
    out_size=np.shape(samples[0])[0],
    layer_sizes=[data_store.config["memory_size"]])
#lstm.load("lstm_wordset_save")

lstm.decode = data_store.decode_char_list
lstm.encode = data_store.encode_char_list

lstm.train(
        samples,
        seq_length=data_store.config["sequence_length"],
        iterations=config["iterations"],
        target_loss=config["target_loss"],
        learning_rate=config["learning_rate"],
        save_dir=os.path.join(os.getcwd(), "lstm_loremipsum_save"),
        dry_run=False)
