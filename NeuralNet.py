import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate, activation_f):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation_f

        self.neural_net = []
        self.biasses = []
        for i in (range(len(layers) - 1)):
            #layer = np.random.random((layers[i+1], layers[i]))
            #bias = np.random.random((layers[i+1], 1))
            layer = np.ones((layers[i+1], layers[i]))
            bias = np.ones((layers[i+1], 1))
            self.neural_net.append(layer)
            self.biasses.append(bias)

    def predict(self,input_data):
        inputs = np.asarray(input_data).reshape(len(input_data), 1)
        for layer,bias in zip(self.neural_net, self.biasses):
            result = np.dot(layer, inputs)
            result += bias
            for item in np.nditer(result, op_flags=['readwrite']):
                item[...] = self.activation(item)
            inputs = result
        return result    

def sigmoid(x):
    return 1/(1+np.exp(-x))

