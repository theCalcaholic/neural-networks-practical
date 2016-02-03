import os
import numpy as np
from lstm_network.LSTMLayer import LSTMLayer
from neural_network import NeuralLayer, CachingNeuralLayer


class LSTMNetwork:
    """Class to manage network of multiple lstm layers and output layer"""

    def __init__(self):
        """initialize network"""
        # loss function used for output (not training)
        self.loss_function = LSTMNetwork.euclidean_loss_function
        # 'populated' state of network (see method populate)
        self.populated = False
        # 'trained' state of network (see method train)
        self.trained = False
        # list of layers used for training
        self.training_layers = []
        # list of layers used for prediction (they share weights with training layers but not state/cache)
        self.prediction_layers = []
        # output layer
        self.output_layer = None
        # configuration holder
        self.config = {
            "learning_rate": 0.001,
            "verbose": False,  # whether to print status during training or not
            "time_steps": 1,
            "status_frequency": 10,  # how frequent status shall be printed during training if 'verbose' is True
            "save_dir": os.path.join(os.getcwd(), "lstm_save")  # path to save/load network weights/biases to/from
        }
        # dynamically exchangable function for printing status
        self.get_status = LSTMNetwork.get_status

    def populate(self, in_size, out_size, memory_sizes=None):
        """
        initializes the layers according to size parameters
        @in_size input size for first lstm layer
        @out_size output size for output layer
        @memory_sizes sizes for state vectors of lstm layers
        """

        # create lstm layer which receives input vectors
        self.training_layers.append(LSTMLayer(
            in_size=in_size,
            memory_size=memory_sizes[0]
        ))
        # for each size in memory_sizes create another lstm layer
        for memory_size in memory_sizes[1:]:
            self.training_layers.append(LSTMLayer(
                in_size=self.training_layers[-1].size,  # input size equals output(=memory) size of preceding layer
                memory_size=memory_size
            ))
        # create ouput layer
        self.output_layer = CachingNeuralLayer(
            in_size=memory_sizes[-1],  # input size corresponds to output size of last lstm layer
            out_size=out_size,
            activation_fn=NeuralLayer.activation_linear,
            activation_fn_deriv=NeuralLayer.activation_linear_deriv
        )
        # add output layer to training_layers list
        self.training_layers.append(self.output_layer)
        # add copy of all lstm layers to prediction_layers list using shared weights and biases
        for layer in self.training_layers[:-1]:
            self.prediction_layers.append(LSTMLayer.get_convolutional_layer(
                layer
            ))
        # disable learn function in prediction layers (just to avoid mistakes)
        for layer in self.prediction_layers:
            layer.learn = None
        # add copy of output layer to prediction layers using shared weights and biases
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
        # mark network as populated
        self.populated = True

    def feedforward(self, layers, inputs):
        """calculates output for given inputs using specified list of layers"""
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        outputs = []
        # repeat for all input vectors
        for data in inputs:
            #  feedforward through all layers
            for layer in layers:
                data = layer.feed(
                        input_data=data,  # data will first be the input vector than the output vector of the last layer
                        time_steps=self.config["time_steps"]  # specifies how many inputs should be cached for learning
                    )
            # add output vector to outputs list
            outputs.append(data)

        return outputs

    def learn(self, targets):
        """
        Calculates the weight updates depending on the loss function derived to the individual weights.
        Requires feedforwarding to be finished for backpropagation through time.
        @targets expected outputs to be compared to actual outputs
        """
        output_losses = []
        # retrieve first cache of output layer
        cache = self.output_layer.first_cache.successor
        # initialize pending weight updates for output layer
        pending_output_weight_updates = np.zeros((self.output_layer.size, self.output_layer.in_size))
        # initialize cumulative error vector (output error)
        error = np.zeros((self.output_layer.size, 1))
        for target in targets:
            # collect output losses for backpropagation into lstm layers
            output_losses.append(cache.output_values - target)
            # calculate output weight update resulting from loss of current time step using
            pending_output_weight_updates += np.outer(output_losses[-1], cache.input_values)
            # add current loss to cumulative error
            error += self.loss_function(cache.output_values, target)
            # select next cache (as saved during forward propagation; have a look inside CachingNeuralLayer class)
            cache = cache.successor
            if cache.is_last_cache:
                break

        # apply weight updates to ouput layer
        self.output_layer.weights -= pending_output_weight_updates * self.config["learning_rate"]
        # clip weights matrix to prevent exploding gradient
        self.output_layer.weights = np.clip(self.output_layer.weights, -5, 5)

        # calculate output layer deltas from losses
        deltas = [
            np.dot(self.output_layer.weights.T, output_loss)
            for output_loss in output_losses]

        # iterate over lstm layers (in reversed order)
        for layer in reversed(self.training_layers[:-1]):
            # invoke learn method of each lstm layer with deltas of layer before (beginning with output layer deltas)
            deltas = layer.learn(deltas, self.config["learning_rate"])

        # return cumulative output error
        return error

    def train(self, sequences, iterations=100, target_loss=0, iteration_start=0, dry_run=False):
        """
        trains the net on the specified data
        @sequences a list (or generator object) of training batches
        @ iterations maximum number of iterations. Will terminated training after reaching <iterations> iterations
        @ target_loss After reaching target loss, training will be terminated
        @ iteration_start integer to start counting iterations from (usefull for loading and continuing training)
        @ dry_run if set to True no learning will be applied
        """
        loss_diff = 0
        # visualize weight matrices
        self.visualize("visualize")

        if self.config["verbose"]:
            print("Start training of network.")
        raw_input("Press Enter to proceed...")

        for i in xrange(iteration_start, iterations):
            loss_list = []
            output_list = []
            target_list = []

            for sequence in sequences:
                # since we try to predict the next character, target values equal inputs shifted by 1
                inputs = sequence[:-1]
                targets = sequence[1:]

                # calculate outputs using the training layers
                outputs = self.feedforward(self.training_layers, inputs)
                # learn based on the given targets
                loss = self.learn(targets)

                # collect output and target values for status printing
                output_list.extend(outputs)
                target_list.extend(targets)
                loss_list.append(loss)
            # calculate average loss of last iteration
            iteration_loss = np.average(loss_list)

            # each <status_frequency> iterations print status (if verbose = True) and create visualization of weights
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
            # check for accuracy condition
            if iteration_loss < target_loss:
                print("Target loss reached! Training finished.")
                for layer in self.training_layers:
                    layer.clear_cache()
                self.trained = True
                return
        # mark network as trained
        self.trained = True

    def reroll_weights(self):
        """randomize weights and biases"""
        self.populate(
            in_size=self.training_layers[0].in_size,
            out_size=self.output_layer.size,
            memory_sizes=[layer.size for layer in self.training_layers])

    def predict(self, input_vector):
        """calculate output for input vector using prediction layers"""
        outputs = self.feedforward(self.prediction_layers, [input_vector])
        return outputs[-1]

    def save(self, path=None):
        """save weights and biases to directory"""
        if path is None:
            base_path = os.path.join(os.getcwd(), self.config["save_dir"])
        else:
            base_path = path
        for idx, layer in enumerate(self.training_layers[:-1]):
            layer.save(os.path.join(base_path, "layer_" + str(idx)))
        self.training_layers[-1].save(os.path.join(base_path, "output_layer.npz"))

    def load(self, path=None):
        """load weights and biases from directory"""
        if path is None:
            base_path = os.path.join(os.getcwd(), self.config["save_dir"])
        else:
            base_path = path
        for idx, layer in enumerate(self.training_layers[:-1]):
            layer.load(os.path.join(base_path, "layer_" + str(idx)))
        self.training_layers[-1].load(os.path.join(base_path, "output_layer.npz"))

    @classmethod
    def get_status(self, output_list, target_list, iteration="", learning_rate="", loss="", loss_difference=""):
        """default status string generating method (usually overwritten)"""
        status = \
            "Iteration: " + iteration + "\n" + \
            "Learning rate: " + learning_rate + "\n" + \
            "loss: " + loss + " (" + \
                (loss_difference if loss_difference[0] == "-" else "+" + loss_difference) + ")\n" + \
            "Target: " + str(target_list) + "\n" \
            "Output: " + str(output_list) + "\n\n"
        return status

    def freestyle(self, seed, length):
        """predict a sequence of <length> outputs, starting with <seed> as input"""
        freestyle = [seed]
        for i in range(length):
            input_vector = np.array(freestyle[-1])
            freestyle.append(self.predict(input_vector))
        return freestyle

    def configure(self, config):
        """update network configuration from dictionary <config>"""
        for key in set(self.config.keys()) & set(config.keys()):
            self.config[key] = config[key]

    @classmethod
    def euclidean_loss_function(cls, predicted, target):
        """returns squared error"""
        return (predicted - target) ** 2

    def visualize(self, path):
        """generate visualization of weights and biases"""
        for idx, layer in enumerate(self.training_layers[:-1]):
            layer.visualize(path, idx + 1)
        self.output_layer.visualize(os.path.join(path, "obs_OutputL_1_0.pgm"))
