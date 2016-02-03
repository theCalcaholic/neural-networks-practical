import numpy as np
import random


class DataStore(object):
    """stores and offers easy access to training data for neural networks (for now it's character based only)"""
    def __init__(self):
        """initialize DataStore object"""
        self.raw_input = None
        self.input_data = None
        self.data_set = None  # contains every single encodable character (once)
        self.data_to_int = None  # dictionary to quickly convert index of highest data vector position to ord(character)
        self.int_to_data = None  # dicitionary to quickly convert ord(character) to index of highest data vector
        self.generated = False  # flag to save initialization state
        self.in_size = None
        self.out_size = None
        # config dictionary to store configuration options
        self.config = {
            'memory_size': None,  # memory size - only needed if extend_alphabet is True
            'time_depth': None,  # number of steps in time to learn
            'extend_alphabet': False,
            'sequence_length': None  # size of training batches
        }

    def load_file(self, path_to_textfile):
        """load (ascii) file content from path and set as training input"""
        self.generated = False
        self.raw_input = open(path_to_textfile, 'r').read()

    def set_input_text(self, text):
        """set training input to <text>"""
        self.generated = False
        self.raw_input = text

    def extend_data_set(self):
        """Deprecated! extends data set to the size of the lstm's memory"""
        self.data_set = self.data_set.union(set(
            [(x + 256)
             for x in range(max(0, self.config["memory_size"] - len(self.data_set)))]
        ))

    def length(self):
        """returns the length of the data_set"""
        return len(self.data_set)

    def decode_char(self, data):
        """decode input vector to character"""
        char = self.int_to_data[np.argmax(data)]
        return "#" if char > 255 else chr(char)

    def decode_char_list(self, data_list):
        """decode list of input vectors to list of chars (i.e. string)"""
        decoded = ""
        for dat in data_list:
            decoded += self.decode_char(dat)
        return decoded

    def encode_char(self, char):
        """encode character to input vector (using 1-of-k-folding)"""
        encoded = np.zeros((len(self.data_set), 1))
        encoded[self.data_to_int[ord(char)]] = 1
        return encoded

    def encode_char_list(self, string_in):
        """encode list of characters (string) to list of input vectors"""
        encoded_data = []
        for i in xrange(len(string_in)):
            # encode character in 1-of-k representation
            encoded_data.append(self.encode_char(string_in[i]))
        return encoded_data

    def samples(self):
        """returns generator object for serving sequences of input vectors"""
        # initialize data store
        self.generate_data()
        # create and return generator object
        return Sequences(
            data=self.raw_input,  # training data as string
            sequence_length=self.config['sequence_length'],  # length of single training batch
            encode=self.encode_char_list   # method for encoding strings
        )

    def random_samples(self, amount):
        """returns generator object for serving random sequences of input vectors out of the training data"""
        # initialize data store
        self.generate_data()
        # create and return generator object
        return RandomSequences(
            amount=amount,  # amount of sequences to serve
            data=self.raw_input,  # training data as string
            sequence_length=self.config['sequence_length'],  # length of single training batch
            encode=self.encode_char_list  # method for encoding strings
        )

    def generate_data(self):
        """initialize data set and encode/decode methods"""
        if self.generated:
            return
        # replace newlines by | (for better output to console)
        self.raw_input = self.raw_input.replace("\n", "|")
        # create data_set
        self.data_set = set([ord(x) for x in self.raw_input])
        # extend data set if option set to True
        if self.config["extend_alphabet"]:
            self.extend_data_set()
        # create conversion dictionaries
        self.data_to_int = {dat: idx for idx, dat in enumerate(self.data_set)}
        self.int_to_data = {idx: dat for idx, dat in enumerate(self.data_set)}
        # calculate vector sizes
        self.in_size = len(self.data_set)
        self.out_size = len(self.data_set)
        # mark data store as initialized
        self.generated = True

    def configure(self, config):
        """update data store configuration from dictionary <config>"""
        for key in set(self.config.keys()) & set(config.keys()):
            self.config[key] = config[key]


class Sequences(object):
    """generator class for serving sequences of training data"""
    def __init__(self, data, sequence_length, encode):
        """initialize generator object"""
        self.sequence_length = sequence_length
        self.raw_data = data
        self.current = 0
        self.encode = encode

    def __iter__(self):
        """is iterable"""
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """get next sequence"""
        # retrieve next sequence
        sequence = self[self.current]
        # stop iteration and reset if no sequence left
        if sequence is None:
            self.current = 0
            raise StopIteration()
        # increase counter and return sequence
        self.current += 1
        return sequence

    def __getitem__(self, idx):
        """return item at position <idx>"""
        # calculate slicing start position
        start = idx * self.sequence_length
        # calculate slicing stop position
        stop = min(start + self.sequence_length + 1, len(self.raw_data))

        if start > len(self.raw_data):
            return None

        # return encoded sequence
        return self.encode(self.raw_data[start:stop])


class RandomSequences(object):
    """generator class for serving sequences of training data in random order"""
    def __init__(self, amount, data, sequence_length, encode):
        """
        intialize generator object
        @amount the amount of sequences to serve
        @data the training data
        @sequence_length the lenght of 1 training batch
        @encode function to encode character as data_vector
        """
        self.raw_data = data
        self.current = 0
        self.encode = encode
        self.max = amount
        self.sequence_length = sequence_length
        self.data_length = len(self.raw_data)
        # calculate max_id to startslicing sequence from
        max_id = self.data_length / self.sequence_length
        if not self.data_length % self.sequence_length:
            max_id -= 1
        # create list of available_ids
        self.available_ids = range(max_id)
        # copy available ids to id list and shuffle it
        self.ids = self.available_ids[:]
        random.shuffle(self.ids)

    def __iter__(self):
        """is iterable"""
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """returns next random sequence"""
        # increase counter
        self.current += 1
        # if counter reached maximum, reset and terminate iteration
        if self.current == self.max:
            self.current = 0
            self.ids = self.available_ids[:]
            random.shuffle(self.ids)
            raise StopIteration()
        # get new shuffled id list if it is empty
        if len(self.ids) == 0:
            self.ids = self.available_ids[:]
            random.shuffle(self.ids)
        # calculate start and stop ids for slicing
        rand_id = self.ids.pop(0)
        stop_id = min(rand_id + self.sequence_length + 1, self.data_length - 1)
        # return sequence
        return self.encode(self.raw_data[rand_id:stop_id])

    def __getitem__(self, idx):
        """
        get item at position <idx>
        ATTENTION! Yields undefined behaviour if invoked after a sequence has already been retrieved
        """
        idx -= self.current
        if idx < 0 or idx >= len(self.ids):
            raise IndexError()
        rand_id = self.ids[idx]
        stop_id = min(rand_id + self.sequence_length, self.data_length - 1)
        return self.encode(self.raw_data[rand_id:stop_id])


