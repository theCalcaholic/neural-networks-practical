import os
import numpy as np
from neural_network import CachingBiasedNeuralLayer, CachingNeuralLayer
from neural_network.util import Logger


class MLPNetwork:
    """Class to manage network of multiple lstm layers and output layer"""

    def __init__(self):
        """initialize network"""
        # loss function used for output (not training)
        self.loss_function = MLPNetwork.euclidean_loss_function
        # 'populated' state of network (see method populate)
        self.populated = False
        # 'trained' state of network (see method train)
        self.trained = False
        # list of layers used for training
        self.training_layers = []
        # output layer
        self.output_layer = None
        # configuration holder
        self.config = {
            "learning_rate": 0.001,
            "verbose": False,  # whether to print status during training or not
            "status_frequency": 50,  # how frequent status shall be printed during training if 'verbose' is True
            "save_dir": os.path.join(os.getcwd(), "mlp_save")  # path to save/load network weights/biases to/from
        }
        # dynamically exchangable function for printing status
        self.get_status = MLPNetwork.get_status

    def populate(self, in_size, out_layer, hidden_layers=[]):
        """
        initializes the layers according to size parameters
        @in_size input size for first lstm layer
        @out_size output size for output layer
        @memory_sizes sizes for state vectors of lstm layers
        """
        layers = hidden_layers
        layers.append(out_layer)

        self.training_layers.append(CachingNeuralLayer(
            in_size=in_size,  # input size equals output(=memory) size of preceding layer
            out_size=layers[0]["size"],
            activation_fn=layers[0]["fn"],
            activation_fn_deriv=layers[0]["fn_deriv"]
        ))
        # for each size in memory_sizes create another lstm layer
        for i in range(1, len(layers)):
            self.training_layers.append(CachingNeuralLayer(
                in_size=self.training_layers[-1].size,  # input size equals output(=memory) size of preceding layer
                out_size=layers[i]["size"],
                activation_fn=layers[i]["fn"],
                activation_fn_deriv=layers[i]["fn_deriv"]
            ))
        # mark network as populated
        Logger.log(str(len(self.training_layers)) + " layers created.")
        self.populated = True

    def feedforward(self, layers, inputs):
        """calculates output for given inputs using specified list of layers"""
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        data = inputs
        #print("data shape:" + str(np.shape(data)))
        outputs = []
        # repeat for all input vectors
        #for data in inputs:
            #  feedforward through all layers
        for layer in layers:
            data = layer.feed(
                    input_data=data,  # data will first be the input vector than the output vector of the last layer
                )
        # add output vector to outputs list
        outputs.append(data)

        return outputs

    def learn_rec(self, target, layer_id):
        Logger.debug("learn recursive: " + str(layer_id))
        if layer_id > len(self.training_layers):
            raise Exception("invalid layer id!")
        elif layer_id == len(self.training_layers):
            return [target - self.training_layers[-1].caches[0].output_values]
        else:
            last_deltas = self.learn_rec(target, layer_id+1)
        delta = self.training_layers[layer_id].get_delta(
            in_data=self.training_layers[layer_id].caches[0].input_values,
            delta=last_deltas[0],
            predecessor_activation_deriv=self.training_layers[layer_id-1].activation_deriv)
        self.training_layers[layer_id].learn(
            result=self.training_layers[layer_id].caches[0].output_values.T,
            delta=last_deltas[0].T,
            learning_rate=self.config['learning_rate']
        )
        deltas = [delta]
        deltas.extend(last_deltas)
        return deltas

    def learn(self, target):
        """
        Calculates the weight updates depending on the loss function derived to the individual weights.
        Requires feedforwarding to be finished for backpropagation through time.
        @targets expected outputs to be compared to actual outputs
        """
        if not self.populated:
            raise Exception("MLP Network needs to be populated first! Have a look at MLPNetwork.populate().")

        Logger.debug("mlpnetwork:learn")
        Logger.debug("target: " + str(target))
        Logger.debug("result: " + str(self.training_layers[-1].caches[0].output_values))
        Logger.debug("out_size: " + str(self.training_layers[-1].size))
        Logger.debug("out_weights: " + str(self.training_layers[-1].weights))
        # calculate output error
        deltas = [target - self.training_layers[-1].caches[0].output_values]
        rng = range(len(self.training_layers) - 2, -1, -1)
        #Logger.log("range: " + str(rng))
        #Logger.log("reversed range: " + str(reversed(rng)))
        #raw_input("press Enter")
        for layer_id, last_weights in reversed(zip(
            rng,
            [l.weights for l in self.training_layers[1:]]
        )):
            Logger.debug(str(layer_id) + "/" + str(len(self.training_layers) - 2))
            deltas.append(
                self.training_layers[layer_id].get_delta(
                    self.training_layers[layer_id].caches[0].output_values,
                    deltas[-1],
                    self.training_layers[layer_id + 1].weights
                )
            )

        error = deltas[0][0]**2

        Logger.debug("deltas: " + str(deltas))
        for delta, layer in zip(reversed(deltas), self.training_layers):
            layer.learn(layer.caches[0].input_values, delta, self.config["learning_rate"])

        #self.bias_1 = min(0, -(learning_rate * delta1) + self.bias_1)
        #self.bias_2 = min(0, -(learning_rate * delta2) + self.bias_2)

        # returns the absolute error (distance of target output and actual output)
        return error

        """
        deltas = self.learn_rec(target, 0)
        error = deltas[-1]**2

        return error

        output_losses = []
        loss = self.training_layers[-1].first_cache.successor.output_values - targets[-1]
        delta = np.dot(self.training_layers[-1].weights.T, loss)
        # initialize pending weight updates for output layer
        pending_output_weight_updates = np.zeros((self.output_layer.size, self.output_layer.in_size))
        # initialize cumulative error vector (output error)
        error = self.loss_function(self.training_layers[-1].cache.output_values, targets[-1])
        for layer in reversed(self.training_layers[-1]):
            for target in targets:
                delta = layer.learn(delta)
                weight_update = np.outer(loss, layer.first_cache.successor.input_values)

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
        return error"""

    def train(self, inputs, targets, iterations=100, target_loss=0, iteration_start=0, dry_run=False):
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
        #self.visualize("visualize")

        if self.config["verbose"]:
            print("Start training of network.")
        raw_input("Press Enter to proceed...")

        for i in xrange(iteration_start, iterations):
            loss_list = []
            output_list = []
            target_list = []

            for input_value, target_value in zip(inputs, targets):

                # calculate outputs using the training layers
                outputs = self.feedforward(self.training_layers, input_value)
                # learn based on the given targets
                loss = self.learn(target_value)

                # collect output and target values for status printing
                output_list.append(float(outputs[0]))
                target_list.append(float(target_value))
                loss_list.append(loss)
            # calculate average loss of last iteration
            iteration_loss = np.average(loss_list)

            # each <status_frequency> iterations print status (if verbose = True) and create visualization of weights
            if not (i + 1) % self.config["status_frequency"] \
                    or iteration_loss < target_loss:
                """if dry_run:
                    self.load()
                else:
                    self.save()
                self.visualize("visualize")"""
                loss_diff -= iteration_loss
                if self.config["verbose"]:
                    print(self.get_status(
                        list(output_list),
                        list(target_list),
                        iteration=str(i),
                        learning_rate=str(self.config["learning_rate"]),
                        loss=str(iteration_loss),
                        loss_difference=str(-loss_diff)
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
            out_layer={
                "size": self.training_layers[-1].size,
                "fn": self.training_layers[-1].activation_function,
                "fn_deriv": self.training_layers[-1].activation_function_deriv
            },
            hidden_layers=[{
                "size": layer.size,
                "fn": layer.activation_function,
                "fn_deriv": layer.activation_function_deriv
            } for layer in self.training_layers[1:-1]]
        )

    def predict(self, input_vector):
        """calculate output for input vector using prediction layers"""
        outputs = self.feedforward(self.training_layers, input_vector)
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
            "Iteration: " + str(int(iteration) + 1) + "\n" + \
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
