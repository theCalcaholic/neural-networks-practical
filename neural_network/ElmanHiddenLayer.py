from NeuralLayer import NeuralLayer, BiasedNeuralLayer
from LayerCache import LayerCache
import numpy as np


class ElmanHiddenLayer(BiasedNeuralLayer):

    def __init__(self,
                 in_size, out_size,
                 predecessors={}, successors={}, activation_fn=None, activation_fn_deriv=None):
        self.predecessors = {'input': None, 'context': None}
        self.successors = {'output': None, 'context': None}
        self.predecessors = predecessors
        self.successors = successors
        self.cache
        self.in_size = in_size
        self.size = out_size
        self.context_layer = BiasedNeuralLayer(
                in_size=in_size,
                out_size=out_size,
                activation_fn=NeuralLayer.activation_sigmoid,
                activation_fn_deriv=NeuralLayer.activation_sigmoid_deriv)
        self.predecessors['context'] = self.context_layer
        self.successors['context'] = self.context_layer
        if len(predecessors) >= 2 and isinstance(predecessors[1], ElmanHiddenLayer):
            self.weights = predecessors['hidden'].weights
            self.biases = predecessors['hidden'].biases
            self.context_layer.weights = predecessors['hidden'].context_layer.weights
            self.context_layer.biases = predecessors['hidden'].context_layer.biases
        else:
            super(self.__class__).__init__(
                    in_size + out_size,
                    out_size,
                    predecessors,
                    successors,
                    activation_fn or NeuralLayer.activation_sigmoid,
                    activation_fn_deriv or NeuralLayer.activation_sigmoid_deriv)

    def feed(self, data):
        self.cache.output = super(self.__class__).feed(
                np.vstack((
                    data,
                    self.predecessors['context'].cache.output)))
        self.successors['context'].feed(self.cache.output)
        return self.successors['output'].feed(self.cache.output)

    def learn(self, result, delta, learning_rate):
        # TODO: Backpropagation over time
        super(self.__class__).learn(result="", delta="", learning_rate="")

    def unfold(self, depth):
        if depth == 0:
            return

        successor = ElmanHiddenLayer(
                self.in_size,
                self.size,
                predecessors={'input': self.predecessors['input'], 'hidden': self},
                successors={'output': self.successors['output']},
                activation_fn=self.activation,
                activation_fn_deriv=self.activation_deriv)
        self.successors['hidden'] = successor
        self.successors['hidden'].unfold(depth - 1)

    def add_successor(self, successor):
        if isinstance(successor, ElmanHiddenLayer):
            self.successors['hidden'] = successor
        else:
            self.successors['output'] = successor

    def add_predecessor(self, predecessor):
        if isinstance(predecessor, ElmanHiddenLayer):
            self.predecessors['hidden'] = predecessor
        else:
            self.predecessors['input'] = predecessor