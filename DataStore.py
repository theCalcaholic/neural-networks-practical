import numpy as np
import random
#from scimage.io import imread

class DataStore(object):
    def __init__(self):
        self.raw_input = None
        self.input_data = None
        self.data_set = None
        self.data_to_int = None
        self.int_to_data = None
        self.config = {
            'memory_size': None,
            'in_size': None,
            'out_size': None,
            'sequence_length': None,
            'extend_alphabet': False
        }

    def load_file(self, path_to_textfile):
        self.raw_input = open(path_to_textfile, 'r').read()

    #def load_img(self, path_to_img):
    #    self.raw_input = imread(path_to_img, as_grey=True, flatten=True)


    def set_input_text(self, text):
        self.raw_input = [text]

    def extend_data_set(self):
        self.data_set = self.data_set.union(set([(x + 256)
                                            for x in range(max(0, self.config["memory_size"] - len(self.data_set)))]))

    def length(self):
        return len(self.tag_set)

    def decode_char(self, data):
        char = self.int_to_data[np.argmax(data)]
        return "#" if char > 255 else chr(char)

    def decode_char_list(self, data_list):
        decoded = ""
        for dat in data_list:
            decoded += self.decode_char(dat)
        return decoded

    def encode_char(self, char):
        encoded = np.zeros((len(self.data_set), 1))
        encoded[self.data_to_int[ord(char)]] = 1
        return encoded

    def encode_char_list(self, string_in):
        encoded_data = []
        for i in xrange(len(string_in)):
            # encode character in 1-of-k representation
            encoded_data.append(self.encode_char(string_in[i]))
        return encoded_data

    def encode_img(self, img):
        return img

    def decode_img(self, img_list):
        return np.reshape(img_list, self.img_width, self.img_height)

    def get_samples(self):
        self.generate_data()
        return self.encode_char_list(''.join(self.raw_input))

    def generate_data(self):
        self.data_set = set([ord(x) for x in ''.join(self.raw_input)])
        if self.config["extend_alphabet"]:
            self.extend_data_set()
        self.data_to_int = {dat: idx for idx, dat in enumerate(self.data_set)}
        self.int_to_data = {idx: dat for idx, dat in enumerate(self.data_set)}
        self.config["in_size"] = len(self.data_set)
        self.config["out_size"] = len(self.data_set)

    def get_shuffled_samples(self):
        self.generate_data()
        sample_ids = range(len(self.raw_input) - 1)
        random_samples = []
        for i in range(len(self.raw_input) - 1):
            rand_num = random.randint(0, len(sample_ids)-1)
            rand_id = sample_ids.pop(rand_num)
            random_samples.append(self.raw_input[rand_id])
        return self.encode_string(''.join(random_samples))

    def configure(self, config):
        self.config.update(config)

