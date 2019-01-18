import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

        self.neural_net = []
        self.bias = []
        for i in (range(len(layers) - 1)):
            layer = np.ones((layers[i+1], layers[i]))
            bias = np.ones((layers[i+1], 1))
            self.neural_net.append(layer)
            self.bias.append(bias)

            