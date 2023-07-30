import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -0.5, 1.0],
    [-1.5, 2.7, 3.3, -0.8]]

# we will now define the hidden layer
# hidden because we define , optimise. But we will not adjust it, neural network should do it automatically

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5) # n neurons can be anything u want
layer2 = Layer_Dense(5, 2) # we chose 5 because we have 5 neurons in layer1

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)