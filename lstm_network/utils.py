import numpy as np

DEBUG = False

def decode(output_data, int_to_data):
    return int_to_data[np.argmax(output_data)]


def encode(x, data_to_int, noCh):
    input_int = data_to_int[x]
    encoded = np.zeros((noCh, 1))
    encoded[input_int] = 1
    return encoded

def debug(s):
    if DEBUG:
        print(s)

def decode_sequence(l, int_to_data):
    s = ""
    for c in l:
        s += decode(c, int_to_data)
    return s