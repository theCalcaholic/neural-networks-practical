import numpy as np

DEBUG = False

def decode(output_data, int_to_data):
    return int_to_data[np.argmax(output_data)]

def debug(s):
    if DEBUG:
        print(s)
