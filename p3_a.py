inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

some_value = -0.5
weight = 0.7
bias = 0.7

print(some_value*weight) # -0.35
print(some_value + bias) # 0.19999999999999996 (0.2)
# weight  acts upon the magnitude, while the bias offsets the value
# y =mx + b; m is the weight, b is the offset acting towards value X

'''
layer_outputs = [] # initiate a current layer output

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # output of a given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

# zip function basically combines a list of lists, element wise:; 

print(layer_outputs)
'''
# [4.8, 1.21, 2.385]
