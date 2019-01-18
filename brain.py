import NeuralNet as nn

layout = [3,2,2,3]
brain = nn.NeuralNetwork(layout, 0.1)

for layer in brain.neural_net:
    print(layer)

for bias in brain.bias:
    print(bias)