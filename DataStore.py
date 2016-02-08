import numpy as np
import random


class SymmetricDataStore(DataStore):
    pass


class DataStore(object):
    """stores and offers easy access to training data for neural networks (for now it's character based only)"""
    def __init__(self, input_data_set, output_data_set, input_data):
        """initialize DataStore object"""
        self.raw_input = input_data
        self.input_data_set = input_data_set  # contains every single encodable character (once)
        self.output_data_set = output_data_set
        # dictionary to quickly convert index of highest data vector position to ord(character)
        self.output_vecid2data = {idx: data for idx, data in enumerate(self.output_data_set)}
        # dicitionary to quickly convert ord(character) to index of highest data vector
        self.output_data2vecid = {data: idx for idx, data in enumerate(self.output_data_set)}
        # dictionary to quickly convert index of highest data vector position to ord(character)
        self.input_vecid2data = {idx: data for idx, data in enumerate(self.input_data_set)}
        # dicitionary to quickly convert ord(character) to index of highest data vector
        self.input_data2vecid = {data: idx for idx, data in enumerate(self.input_data_set)}
        self.in_size = len(self.input_data_set)
        self.out_size = len(self.output_data_set)
        # config dictionary to store configuration options
        self.config = {
            'sequence_length': None  # size of training batches
        }
        self.generated = True  # flag to save initialization state

    def load_file(self, path_to_textfile):
        """load (ascii) file content from path and set as training input"""
        self.generated = False
        self.raw_input = open(path_to_textfile, 'r').read()

    def set_input(self, input_data, input_data_set):
        """set training input to <text>"""
        self.generated = False
        self.raw_input = input_data

    def set_output_data_set(self, output_data_set):
        self.generated = False
        self.output_data_set = output_data_set

    @classmethod
    def decode(cls, vector, mapping):
        """decode input vector to character"""
        val = mapping[np.argmax(vector)]
        return val

    def decode_output(self, vector):
        return DataStore.decode(vector, self.output_vecid2data)

    def decode_input(self, vector):
        return DataStore.decode(vector, self.input_vecid2data)

    @classmethod
    def decode_all(self, vector_list, mapping):
        """decode list of input vectors to list of chars (i.e. string)"""
        decoded = []
        for vector in vector_list:
            decoded.append(self.decode(vector, mapping))
        return decoded

    def decode_all_inputs(self, vector_list):
        return DataStore.decode_all(vector_list, self.input_vecid2data)

    def decode_all_outputs(self, vector_list):
        return DataStore.decode_all(vector_list, self.output_vecid2data)

    @classmethod
    def encode(cls, value, mapping):
        """encode character to input vector (using 1-of-k-folding)"""
        encoded = np.zeros((len(mapping), 1))
        encoded[mapping[value]] = 1
        return encoded

    def encode_input(self, value):
        return DataStore.encode(value, self.input_data2vecid)

    def encode_output(self, value):
        return DataStore.encode(value, self.output_data2vecid)

    @classmethod
    def encode_all(cls, values_list, mapping):
        """encode list of characters (string) to list of input vectors"""
        encoded_data = []
        for value in values_list:
            # encode character in 1-of-k representation
            encoded_data.append(DataStore.encode(value, mapping))
        return encoded_data

    def encode_all_inputs(self, values_list):
        return DataStore.encode_all(values_list, self.input_data2vecid)

    def encode_all_outputs(self, values_list):
        return DataStore.encode_all(values_list, self.output_data2vecid)

    def samples(self):
        """returns generator object for serving sequences of input vectors"""
        # create and return generator object
        return Sequences(
            data=self.raw_input,  # training data as string
            sequence_length=self.config['sequence_length'],  # length of single training batch
            encode_list=self.encode_all_inputs   # method for encoding strings
        )

    def random_samples(self, amount):
        """returns generator object for serving random sequences of input vectors out of the training data"""
        # create and return generator object
        return RandomSequences(
            amount=amount,  # amount of sequences to serve
            data=self.raw_input,  # training data as string
            sequence_length=self.config['sequence_length'],  # length of single training batch
            encode=self.encode_all_inputs  # method for encoding strings
        )

    def configure(self, config):
        """update data store configuration from dictionary <config>"""
        for key in set(self.config.keys()) & set(config.keys()):
            self.config[key] = config[key]


class Sequences(object):
    """generator class for serving sequences of training data"""
    def __init__(self, data, encode_list, sequence_length):
        """initialize generator object"""
        self.sequence_length = sequence_length
        self.raw_data = data
        self.current = 0
        self.encode = encode_list

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
    def __init__(self, amount, data, sequence_length, encode_list):
        """
        intialize generator object
        @amount the amount of sequences to serve
        @data the training data
        @sequence_length the lenght of 1 training batch
        @encode function to encode character as data_vector
        """
        self.raw_data = data
        self.current = 0
        self.encode = encode_list
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


class DataMapper(object):
    def __init__(self, data_set_in, data_set_out):
        self.mapping = {data_in: data_out for data_in, data_out in zip(data_set_in, data_set_out)}

    def convert(self, data_in):
        return self.mapping[data_in]
