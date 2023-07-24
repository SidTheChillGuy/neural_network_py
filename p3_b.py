import numpy as np
'''
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(inputs, weights) + bias
output2 = np.dot(weights, inputs) + bias

print(output)
print(output2)

# 4.8 for both
'''

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)
# [4.8   1.21  2.385]

output2 = np.dot(inputs, weights) + biases
print(output2)
# doesnt run, a shape error