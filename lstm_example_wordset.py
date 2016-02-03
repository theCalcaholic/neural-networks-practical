import os
import random
import numpy as np
from lstm_network import LSTMNetwork
from DataStore import DataStore
from lstm_network import util


### Example status output after 500 iterations on the training set found in ./lstm_training_data/loremipsum.txt
"""
Iteration 500
Learning rate: 0.01
Loss: 0.0661109480215  /\ 0.00698906546459
Target: orem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur
Output: srem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet, Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet, Lorem ipsum dolor sit amet. consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet, Luis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accussam et iusto odio dignissim qui blandit praesent luptatum .rril delenit augue duis dolore te feugait nulla facilisis aores ipsum dolor sit amet, consettetuer adipiscing elitr sed diam nonumy  eibm euismod tincidunt ut labreet dolore magna aliquam erat volutpat. Lt iisi enim ad minim veniam,  uis nostrud emerci tation ullamcorper suscipit lobortis nisl ut aliquip eo ea commodo consequat. vuis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolor  eu eeugiat nulla facilisis  t vero eros et accussan et iusto odio dignissim qui blandit praesent auptatum vrril del iit augue duis doloreste feugait nulla facilisis aam vite  tempor cim selutatsilis eleit nd ottio oeon  e dllii imper iet dolini ie tuod eanim nlacrrat sacergtosuem assum  Lorem ipsum doloreeeu amet, consettetuer adipiscing elitr sed diam nonumm  nibs euismod tincidunt ut laoreet dolore magna aliquam erat solutpat. lt visi enim ad minim veniam,  uis nostrud exerci tation ullamcorper suscipit loboreis nisl ut aliquip es ea cimmodo consequat. vuis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis  et vero eos et accusam et iusto duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet, Lorem ipsum dolor sit amet. consetetur sadipscing elitr, sed diam nonumm eirmod tempor invidunt ut labore et dolore magna aliquaam erat, sed diam voluptua. At vero eos et accusam et iusto duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet, Lorem ipsum dolor sit amet. consetetur sadipscing elitr, st accusam aliquaam diam niam dolor  solores eio eirmod eos erat, at donumy eer tempor it et iniidunt lusto dabore etet clita kaset gubergren, las  manna nonrebum. sanctus sea ced takimata sa cero soluptua. ett Lorem ipsum dolor sit amet, Lorem ipsum dolor sit amet, consetetur

freestyle: juauee emusuari d iasooeutd cor
"""

config = {
    "memory_size": 200,
    "time_steps": 10,
    "learning_rate": 0.01,
    "iterations": 10000,
    "target_loss": 0.01,
    "use_output_layer": True,
    "verbose": True,
    "status_frequency": 10,
    "save_dir": os.path.join(os.getcwd(), "lstm_loremipsum_save")
}

util.Logger.DEBUG = False

# initialize data store object
data_store = DataStore()


# Training sets
"""input_text = "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"\
             + "abcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddefabcddef"""""
input_text = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, ipsum sed diam"

# Either use set_input_text(string) to use a string as trainings set or define a text file using load_file(file_path)
data_store.set_input_text(input_text)
#data_store.load_file("lstm_training_data/vanishing_vocables_de.txt")
#data_store.load_file("lstm_training_data/loremipsum.txt")


data_store.configure({
    "memory_size": config["memory_size"],
    "sequence_length": config["time_steps"],
    "extend_alphabet": False #only necessary if output layer is disabled
})

# samples is a generator object for sequences of length data_store.config['time_steps']
samples = data_store.samples()

# Uncomment to use random samples for training. The amount specifies the number of sequences
# to be trained during one iteration. After each iteration the sequences will be shuffled again.
#samples = data_store.random_samples(amount=20)

lstm = LSTMNetwork()

lstm.configure(config)
lstm.get_status = util.get_status_function(data_store, lstm, config["status_frequency"])
lstm.populate(
    in_size=np.shape(samples[0][0])[0],
    out_size=np.shape(samples[0][0])[0],
    layer_sizes=[data_store.config["memory_size"]])

# Uncomment to load previously saved network weights and biases from directory
#lstm.load(config["save_dir"])


# train the network
lstm.train(
        sequences=samples,
        iterations=config["iterations"],
        target_loss=config["target_loss"],
        dry_run=False
)

# Generate string by feeding lstm network with it's own output:
character = data_store.int_to_data[random.randint(0, data_store.length() - 1)]
seed = data_store.encode_char(chr(character))
print(data_store.decode_char_list(lstm.freestyle(seed, 100)))