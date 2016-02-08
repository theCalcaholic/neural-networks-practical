import random
import numpy as np


def get_status_function(data_store, lstm, status_frequency):
    def get_status(output_list, target_list, iteration="", learning_rate="", loss="", loss_difference=""):
        loss_difference = float(loss_difference)
        iteration = int(iteration)
        status = "\nIteration " + str(iteration + 1) + "\n" + \
                 "Learning rate: " + learning_rate + "\n" + \
                 "Loss: " + loss + "  " + \
                    ("\\/" if loss_difference > 0 else ("--" if loss_difference == 0 else "/\\")) + \
                    " " + str(loss_difference)[1:] + "\n" + \
                 "Target: " + ''.join([data_store.decode_output(c) for c in target_list]).replace("\n", "|") + "\n" + \
                 "Output: " + ''.join(data_store.decode_all_outputs(output_list)).replace("\n", "|") + "\n\n"
        if not (int(iteration)+1) % (status_frequency * 5):
            character = data_store.input_vecid2data[random.randint(0, len(data_store.input_data_set) - 1)]
            seed = data_store.encode_input(character)
            freestyle = ''.join(data_store.decode_all_outputs(lstm.freestyle(seed, 30)))
            status += "freestyle: " + freestyle + "\n\n"
        return status
    return get_status


class Logger(object):
    DEBUG = False

    def __init__(self):
        raise Exception("This class is abstract. It cannot be initialized!")

    @classmethod
    def log(cls, s):
        print(s)

    @classmethod
    def debug(cls, s):
        if Logger.DEBUG:
            print(s)
