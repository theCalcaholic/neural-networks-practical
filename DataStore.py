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
        self.generated = False
        self.config = {
            'memory_size': None,
            'in_size': None,
            'out_size': None,
            'time_depth': None,
            'extend_alphabet': False,
            'sequence_length': None
        }

    def load_file(self, path_to_textfile):
        self.generated = False
        self.raw_input = open(path_to_textfile, 'r').read()

    def set_input_text(self, text):
        self.generated = False
        self.raw_input = text

    def extend_data_set(self):
        self.data_set = self.data_set.union(set(
            [(x + 256)
             for x in range(max(0, self.config["memory_size"] - len(self.data_set)))]
        ))

    def length(self):
        return len(self.data_set)

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

    def samples(self):
        self.generate_data()
        return Sequences(
            data=self.raw_input,
            sequence_length=self.config['sequence_length'],
            encode=self.encode_char_list
        )

    def random_samples(self, amount):
        self.generate_data()
        return RandomSequences(
            amount=amount,
            data=self.raw_input,
            sequence_length=self.config['sequence_length'],
            encode=self.encode_char_list
        )

    def generate_data(self):
        if self.generated:
            return
        self.raw_input = self.raw_input.replace("\n", "|")
        self.data_set = set([ord(x) for x in self.raw_input])
        if self.config["extend_alphabet"]:
            self.extend_data_set()
        self.data_to_int = {dat: idx for idx, dat in enumerate(self.data_set)}
        self.int_to_data = {idx: dat for idx, dat in enumerate(self.data_set)}
        self.config["in_size"] = len(self.data_set)
        self.config["out_size"] = len(self.data_set)
        self.generated = True

    def configure(self, config):
        for key in set(self.config.keys()) & set(config.keys()):
            self.config[key] = config[key]


class Sequences(object):
    def __init__(self, data, sequence_length, encode):
        self.sequence_length = sequence_length
        self.raw_data = data
        self.current = 0
        self.encode = encode

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        sequence = self[self.current]
        if sequence is None:
            self.current = 0
            raise StopIteration()
        self.current += 1
        return sequence

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        stop = min(start + self.sequence_length + 1, len(self.raw_data))
        if start > len(self.raw_data):
            return None
        return self.encode(self.raw_data[start:stop])


class RandomSequences(object):
    def __init__(self, amount, data, sequence_length, encode):
        self.raw_data = data
        self.current = 0
        self.encode = encode
        self.max = amount
        self.sequence_length = sequence_length
        self.data_length = len(self.raw_data)
        max_id = self.data_length / self.sequence_length
        if not self.data_length % self.sequence_length:
            max_id -= 1
        self.available_ids = range(max_id)
        self.ids = self.available_ids[:]
        random.shuffle(self.ids)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self.current += 1
        if self.current == self.max:
            self.current = 0
            self.ids = self.available_ids[:]
            random.shuffle(self.ids)
            raise StopIteration()
        if len(self.ids) == 0:
            self.ids = self.available_ids[:]
            random.shuffle(self.ids)
        rand_id = self.ids.pop(0)
        stop_id = min(rand_id + self.sequence_length + 1, self.data_length - 1)
        return self.encode(self.raw_data[rand_id:stop_id])

    def __getitem__(self, idx):
        idx = idx - self.current
        if idx < 0 or idx >= len(self.ids):
            raise IndexError()
        rand_id = self.ids[idx]
        stop_id = min(rand_id + self.sequence_length, self.data_length - 1)
        return self.encode(self.raw_data[rand_id:stop_id])


