import NeuralNet as nn
import random

# dataset for the xor problem
input_list = [[1,1,0], [1,0,1], [0,1,1], [0,0,0]]
brain = nn.boring_NN(2,10,1,0.3)
cerebro = nn.NeuralNetwork((2,10,1), 0.3, nn.sigmoid)

for i in range(10000):
    train = random.choice(input_list)
    brain.train(train[:2], train[2])
    cerebro.train(train[:2], train[2])

for test in input_list:
    prediction1 = brain.predict(test[:2])
    prediction2 = cerebro.predict(test[:2])
    '''if prediction[0,0] >= 0.5:
        xor = 1
    else:
        xor = 0'''
    print('Sample {} was predicted as {} by the brain and {} by the cerebro. It actually was {}'.format(test[:2], prediction1,prediction2, test[2]))    