from neural_network.NeuralLayer import NeuralLayer


class HiddenLayer(NeuralLayer):
    def learn(self, result, delta, learning_rate):
        super(HiddenLayer, self).learn(result, delta, learning_rate)
        self.biases = min(0, -(learning_rate * delta) + self.biases)