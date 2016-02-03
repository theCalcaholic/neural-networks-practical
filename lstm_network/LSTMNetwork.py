import os
import numpy as np
from lstm_network.LSTMLayer import LSTMLayer
from neural_network import NeuralLayer, CachingNeuralLayer


class LSTMNetwork:

    def __init__(self):
        self.loss_function = LSTMNetwork.euclidean_loss_function
        self.layers_spec = None
        self.populated = False
        self.trained = False
        self.training_layers = []
        self.prediction_layers = []
        self.output_layer = None
        self.config = {
            "learning_rate": 0.001,
            "verbose": False,
            "use_output_layer": True,
            "time_steps": 1,
            "status_frequency": 10,
            "save_dir": os.path.join(os.getcwd(), "lstm_save")
        }
        self.get_status = LSTMNetwork.get_status

    def populate(self, in_size, out_size, layer_sizes=None):

        self.training_layers.append(LSTMLayer(
            in_size=in_size,
            memory_size=layer_sizes[0]
        ))
        for layer_size in layer_sizes[1:]:
            self.training_layers.append(LSTMLayer(
                in_size=self.training_layers[-1].size,
                memory_size=layer_size
            ))
        self.output_layer = CachingNeuralLayer(
            in_size=layer_sizes[-1],
            out_size=out_size,
            activation_fn=NeuralLayer.activation_linear,
            activation_fn_deriv=NeuralLayer.activation_linear_deriv
        )
        self.training_layers.append(self.output_layer)
        for lstm_layer in self.training_layers[:-1]:
            self.prediction_layers.append(LSTMLayer.get_convolutional_layer(
                lstm_layer
            ))
        for layer in self.prediction_layers:
            layer.learn = None
        self.prediction_layers.append(
            CachingNeuralLayer(
                in_size=self.output_layer.in_size,
                out_size=self.output_layer.size,
                activation_fn=self.output_layer.activation,
                activation_fn_deriv=self.output_layer.activation_deriv
            )
        )
        self.prediction_layers[-1].weights = self.output_layer.weights
        self.prediction_layers[-1].biases = self.output_layer.biases
        self.populated = True

    def feedforward(self, layers, inputs):
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        outputs = []
        for data in inputs:

            for layer in layers:
                data = layer.feed(
                        input_data=data,
                        time_steps=self.config["time_steps"]
                    )
            outputs.append(data)

        return outputs

    def learn(self, targets):
        output_losses = []
        cache = self.output_layer.first_cache.successor
        pending_output_weight_updates = np.zeros((self.output_layer.size, self.output_layer.in_size))
        error = np.zeros((self.output_layer.size, 1))
        for target in targets:
            output_losses.append(cache.output_values - target)
            pending_output_weight_updates += np.outer(output_losses[-1], cache.input_values)
            error += self.loss_function(cache.output_values, target)
            cache = cache.successor
            if cache.is_last_cache:
                break
        deltas = [
            np.dot(self.output_layer.weights.T, output_loss)
            for output_loss in output_losses]

        #print("deltas 3: " + str(np.shape(deltas)))
        self.output_layer.weights -= pending_output_weight_updates * self.config["learning_rate"]

        for layer in reversed(self.training_layers[:-1]):
            deltas = layer.learn(deltas, self.config["learning_rate"])
            #print("deltas x: " + str(np.shape(deltas)))

        return error

    def train(self, sequences, iterations=100, target_loss=0, iteration_start=0, dry_run=False):
        loss_diff = 0
        self.visualize("visualize")

        if self.config["verbose"]:
            print("Start training of network.")
        raw_input("Press Enter to proceed...")

        for i in xrange(iteration_start, iterations):
            loss_list = []
            output_list = []
            target_list = []

            for sequence in sequences:
                inputs = sequence[:-1]
                targets = sequence[1:]

                outputs = self.feedforward(self.training_layers, inputs)
                loss = self.learn(targets)

                output_list.extend(outputs)
                target_list.extend(targets)
                loss_list.append(loss)
            iteration_loss = np.average(loss_list)
            #print(str(iteration_loss))

            if not (i + 1) % self.config["status_frequency"] \
                    or iteration_loss < target_loss:
                if dry_run:
                    self.load()
                else:
                    self.save()
                self.visualize("visualize")
                loss_diff -= iteration_loss
                if self.config["verbose"]:
                    print(self.get_status(
                        output_list,
                        target_list,
                        iteration=str(i),
                        learning_rate=str(self.config["learning_rate"]),
                        loss=str(iteration_loss),
                        loss_difference=str(loss_diff)
                    ))
                loss_diff = iteration_loss
                if iteration_loss < target_loss:
                    print("Target loss reached! Training finished.")
                    for layer in self.training_layers:
                        layer.clear_cache()
                    self.trained = True
                    return
        self.trained = True

    def reroll_weights(self):
        self.populate(
            in_size=self.training_layers[0].in_size,
            out_size=self.output_layer.size,
            layer_sizes=[layer.size for layer in self.training_layers])

    def predict(self, input_vector):

        outputs = self.feedforward(self.prediction_layers, [input_vector])
        return outputs[-1]

    def save(self, path=None):
        if path is None:
            base_path = os.path.join(os.getcwd(), self.config["save_dir"])
        else:
            base_path = path
        for idx, layer in enumerate(self.training_layers[:-1]):
            layer.save(os.path.join(base_path, "layer_" + str(idx)))
        self.training_layers[-1].save(os.path.join(base_path, "output_layer.npz"))

    def load(self, path=None):
        if path is None:
            base_path = os.path.join(os.getcwd(), self.config["save_dir"])
        else:
            base_path = path
        for idx, layer in enumerate(self.training_layers[:-1]):
            layer.load(os.path.join(base_path, "layer_" + str(idx)))
        self.training_layers[-1].load(os.path.join(base_path, "output_layer.npz"))

    @classmethod
    def get_status(self, output_list, target_list, iteration="", learning_rate="", loss="", loss_difference=""):
        status = \
            "Iteration: " + iteration + "\n" + \
            "Learning rate: " + learning_rate + "\n" + \
            "loss: " + loss + " (" + \
                (loss_difference if loss_difference[0] == "-" else "+" + loss_difference) + ")\n" + \
            "Target: " + str(target_list) + "\n" \
            "Output: " + str(output_list) + "\n\n"
        return status

    def freestyle(self, seed, length):
        freestyle = [seed]
        for i in range(length):
            input_vector = np.array(freestyle[-1])
            freestyle.append(self.predict(input_vector))
        return freestyle

    def configure(self, config):
        for key in set(self.config.keys()) & set(config.keys()):
            self.config[key] = config[key]

    @classmethod
    def euclidean_loss_function(cls, predicted, target):
        return (predicted - target) ** 2

    def visualize(self, path):
        for idx, layer in enumerate(self.training_layers[:-1]):
            layer.visualize(path, idx + 1)
        self.output_layer.visualize(os.path.join(path, "obs_OutputL_1_0.pgm"))
