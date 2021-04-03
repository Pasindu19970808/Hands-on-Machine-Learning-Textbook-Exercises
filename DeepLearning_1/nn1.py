#a neural network from scratch
import numpy as np
import math

#rather than sending one example at a time, sending in a batch allows to generalize. When we send one at a time it will try to fit each. However sending a batch generalizes it. What happens when we send one at a time is that the neuron will just bounce around.
#However showing all the samples at the time results in overfitting and not enough generalization
inputs = [[1,2,3,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]]

#layer 1      
weights_layer1 = [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]

#bias weight
biases_layer1 = [2,3,0.5]


#layer 2
weights_layer2 = [[0.1,-0.14,0.5],
            [-0.5,0.12,-0.33],
            [-0.44,0.73,-0.13]]

#bias weight
biases_layer2 = [-1,2,-0.5]

""" layer1_outputs = np.dot(inputs, np.transpose(weights_layer1)) + biases_layer1
layer2_outputs = np.dot(layer1_outputs, np.transpose(weights_layer2)) + biases_layer2 """


def dotprod(layer_weights,layer_biases,layer_inputs):
    input_weight =  [[(input_1,weight) for weight in layer_weights] for input_1 in layer_inputs]

    output_layer = list(map(lambda o: list(map(lambda p: sum(p),o)),list(map(lambda q: tuple(zip(*(layer_biases,q))),list(map(lambda r: list(map(lambda s: sum(s),r)),list(map(lambda t: list(map(lambda u: list(map(lambda v: math.prod(v),u)),t)),list(map(lambda w: list(map(lambda x: tuple(zip(*(x[0],x[1]))),w)),input_weight)))))))))) 
    return output_layer

""" test = np.array([[[[1,2,3],
        [1,3,4]],
        [[1,2,3],
        [1,3,4]],
        [[1,2,3],
        [1,3,4]]],
        [[[1,2,3],
        [1,3,4]],
        [[1,2,3],
        [1,3,4]],
        [[1,2,3],
        [1,3,4]]]]) """


layer2_outputs = dotprod(weights_layer2,biases_layer2,dotprod(weights_layer1,biases_layer1,inputs))
print(layer2_outputs)