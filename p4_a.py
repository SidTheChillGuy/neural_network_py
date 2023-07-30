# sending inputs as batches

import numpy as np
'''
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -0.5, 1.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# output = np.dot(weights, inputs) + biases
# dot attempts to do 4 rows of input(single list) to 3 coloumns of weights (one per list) and fails.
# we can migitate this using transpose

output = np.dot(inputs, np.array(weights).T) + biases # T for transpose
print(output)

[[ 4.8    1.21   2.385]
 [ 7.65  -1.18  -0.585]
 [ 1.41   1.051  0.026]]
'''

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -0.5, 1.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biases

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(layer2_output)
'''
[[ 0.5031  -1.04185 -2.03875]
 [-0.3623  -1.77355 -4.65135]
 [-0.99314  1.41254 -0.35655]]
'''