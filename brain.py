import NeuralNet as nn

layout = [2,2,1]
brain = nn.NeuralNetwork(layout, 0.1,nn.sigmoid)

inputs = [1,1]

print(brain.predict(inputs))