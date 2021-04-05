#a neural network from scratch
import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data

#this only initializes the seed in numpy s that all data outputs remain consistent. Also we will use this to import datasets
nnfs.init()

#region
#EXPLANATIONS

#rather than sending one example at a time, sending in a batch allows to generalize. When we send one at a time it will try to fit each. However sending a batch generalizes it. What happens when we send one at a time is that the neuron will just bounce around.
#However showing all the samples at the time results in overfitting and not enough generalization

#we call it hidden layer because we do not specify how that layer changes

#How do we initialize layers:
#- trained model that we want to load. When we save a model we are only saving the weights and biases
#- when we load the model we set the weights and biases. 

#Making a new neural network
#- Initializing weights: As random values between -1 and +1. Always better to have a smaller and tighter range. We prefer having small values for our weights and biases. 
#- When you have larger weights and biases, what happens is that as the value goes through the network, at each neuron, the value increases and ultimately it explodes. 
#- a good starting point for weights is -0.1 and +0.1
#- for biases we tend to initialize these as zeros(However if you bias is zero, and your neuron weights are too small, causing your output also to be zero as bias is zero, it results in a zero output. Zero times any weight is zero, and if that neurons bias is also zero, we have another zero output, resulting in all zeros causing the network to die)
#- for initialization we use np.rand.randn(shape_of_return_array)

#Step Functions
#Heaviside Step Function: 
#If the result of weight*input is greater than 0, then the output is one. If its less than 0, the output is 0. 
#The result of the weight*input is what is being fed through the activation function. 
#Every neuron in the hidden layers and output layers has an activation function. 

#Sigmoid Activation Function
#Provides a more granular output. Not just a 1 or 0 like Heaviside Step Function.
#The sigmoid function makes things much better when calculating the loss, as we know how close we were to calculating a 0 or a 1. While a Heaviside step function does not give that.

#Rectified Linear Unit
#If (weights*input + bias)>0, then the value is x. If the value is greater than 0, then the output is (weights*input + bias)
#This activation function shows how weights can flip the result and bias can offset the result

#Signmoid function has the issue that it has a vanishing gradient(explained later)
#Rectified Linear Unit(Why we use it often)
#- Its fast(very simple calculation)
#- It has required granularity

#Why using Activation Function?
#Can we tune a neural network using weights and biases alone?
#- If we use a linear activation function(weights and biases only), our output will always be linear 
#- This means everything will be a linear output
#- Thus we can only fit linear data and not get any complex pattern
#- Example: if we try to fix a linear activation function to a non linear function like a sine function,  it simple cant fit
#- A ReLU function however can fit a non linear function

#Why does this non linear activation function actually work?
#- bias of a neuron can change the activation/deactivation point of a ReLU activation function
#- the sign of the weight can determine of its an activation or deactivation function
#- Different neurons will have different weights and biases assigned to them and each piece will activate and deactivate along the function. 

#Softmax Activation Function
#To train a model, we need to determine how wrong a model is and how far away a result is from the required value. 
#if we were to use ReLU to give us the answer of the probability of a class, we would only get 0 or 1(whether the input is -10,-9000 or 1000)
#Hence we need to apply the exponent on each output from the final hidden layer at each output neuron
#Then each value will be divided by the sum of the exponents and then we have the probability for each value
#For each input you will have outputs equal to number of the classes which needs to be applied with the exponent normalization
#An issue with the exponential function is that as the input grows, it results in explosion. Also it can result in memory overflow
#One way to combat this explosion is to take the values which are the output of the output layer and substract the maximum value from the values. 
#Substracting the largest value results the largest value being a zero and the rest being negative. 
#Now all your outputs are going to be between 0 and 1. So we dont have to worry about overflow
#Finally the actual normalized output will be identically the same and we have prevented a memory overflow
#For random initializations, your output probabilities will be 1/no.ofclasses

#Calculating the Loss Entropy
#By using only accuracy to calculate the loss, we are throwing away a lot of information
#Showing how wrong or correct we are is important to the optimizer
#MAE(mean absolute error)
#The loss function of choice is categorical cross entropy
#This is more popular for back propagation steps. 
#By log we mean natural log
#Given that we multiply the entire prediction matrix with the one hot encoded matrix, it is essentialy summ of -log(prediction for respective correct class)
#When the probability from softmax is higher, the loss is lower
#When the probability from softmax is lower, the loss is higher

#endregion

""" np.random.seed(0) 
X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]] """

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        #when we have our weights in this shape, we dont have to do transpose
        self.weights = 0.1*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros(shape = (1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        self.exp_values = np.exp(inputs - np.max(inputs,axis = 1, keepdims=True))
        self.output = self.exp_values/np.sum(self.exp_values,axis = 1, keepdims=True)

#first layer of neurons
#layer1 = Layer_Dense(n_inputs = 2, n_neurons = 5)
#takes the output of the layer and produce the activation for the entire layer

""" def dotprod(layer_weights,layer_biases,layer_inputs):
    input_weight =  [[(input_1,weight) for weight in layer_weights] for input_1 in layer_inputs]

    output_layer = list(map(lambda o: list(map(lambda p: sum(p),o)),list(map(lambda q: tuple(zip(*(layer_biases,q))),list(map(lambda r: list(map(lambda s: sum(s),r)),list(map(lambda t: list(map(lambda u: list(map(lambda v: math.prod(v),u)),t)),list(map(lambda w: list(map(lambda x: tuple(zip(*(x[0],x[1]))),w)),input_weight)))))))))) 
    return output_layer  """

""" layer2_outputs = dotprod(weights_layer2,biases_layer2,dotprod(weights_layer1,biases_layer1,inputs)) """


""" layer1 = Layer_Dense(n_inputs = 4,n_neurons = 5)
#output from layer1 is input to layer2
layer2 = Layer_Dense(n_inputs = 5,n_neurons = 2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output) """

#100 data points with 3 different classes 
X,y = spiral_data(samples = 100,classes = 3)

dense1 = Layer_Dense(n_inputs = X.shape[1],n_neurons = 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(n_inputs = 3, n_neurons = 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

#print(activation2.output)

softmax_output = [0.7,0.1,0.2]

target_class = 0

one_hot = np.zeros(shape = (1,3))
one_hot[:,target_class] = 1


print(-math.log(softmax_output*one_hot))

