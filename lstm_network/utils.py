import numpy as np

def decode(output_data, int_to_data):
    return int_to_data[np.argmax(output_data)]