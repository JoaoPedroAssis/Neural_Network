import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate, activation_f):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation_f

        self.neural_net = []
        self.biasses = []
        self.layer_outputs = []
        
        for i in (range(len(layers) - 1)):
            # Other way to initialize the weights (using a normal distribution)
            # layer = np.random.normal(0.0, pow(layers[i], -0.5,(layers[i+1], layers[i])))
            layer = (np.random.random((layers[i+1], layers[i])) - 0.5)
            bias = np.random.random((layers[i+1], 1))
            self.neural_net.append(layer)
            self.biasses.append(bias)

    def predict(self, inputs, train=False):
        '''Multiply the input matrix by the weights matrix
        resulting in the feed forward algorithm. If the 
        training flag is set to True, the function saves the
        outputs of each layer'''

        # inputs = np.asarray(inputs).reshape(len(inputs), 1)
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs, ndmin = 2).T

        for layer,bias in zip(self.neural_net, self.biasses):
            result = np.dot(layer, inputs)
            result += bias
            result = self.activation(result)
            if train == True:
                self.layer_outputs.append(result)
            # The result of a layer is the input for the next one
            inputs = result
        return result

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin = 2).T
        targets = np.array(targets, ndmin = 2).T

        # Clear the list
        self.layer_outputs = []
        self.layer_outputs.append(inputs)
        outs = np.asarray(self.layer_outputs)
        
        net_output = self.predict(inputs, train=True)
        net_errors = [targets - net_output]
    

        previous_error = net_errors[0]
        for layer in reversed(self.neural_net):
            error = np.dot(layer.T,previous_error)
            net_errors.append(error)
            previous_error = error

        for layer,error,i in zip(reversed(self.neural_net), net_errors, reversed(range(len(outs)-1))):
            layer += self.learning_rate * np.dot((error*d_sigmoid(outs[i])), np.transpose(outs[i-1]))
            

 
def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return x*(1 - x)


class boring_NN:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = sigmoid
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass

    
    # query the neural network
    def predict(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs