"""
FULL DATASET: https://pjreddie.com/projects/mnist-in-csv/
"""

import numpy as np 
import scipy.special 
import matplotlib.pyplot as plt 


# loading data files and creating list from it (it's not full training dataset, so score will be low) 
training_data_file = open("datafolder/mnist_train_100.csv") 
train_list = training_data_file.readlines() 
training_data_file.close()

# loading data for testing neural network  
testing_data_file = open("datafolder/mnist_test_10.csv")
test_list = testing_data_file.readlines() 
testing_data_file.close() 



class neuralNetwork(object): 
    def __init__(self, input, hidden, output, learningRate): 
        # number of nodes in each layer 
        self.input = input 
        self.hidden = hidden 
        self.output = output 
        
        # learning rate 
        self.lr = learningRate 

        #activation function 
        self.activation_function = lambda x: scipy.special.expit(x) 

        # number of weight links for layers from input to hidden 
        self.wih = np.random.normal(0.0, pow(self.hidden, -0.5), 
                                    (self.hidden, self.input))
        
        # number of weight links for layers from hidden to output 
        self.who = np.random.normal(0.0, pow(self.output, 0.5), 
                                    (self.output, self.hidden))


    
        
    # training from data 
    def train(self, inputs_list, targets_list): 
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T


        hidden_nodes = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_nodes) 
        final_inputs = np.dot(self.who, hidden_outputs) 
        final_outputs = self.activation_function(final_inputs)


        # output error target - output 
        outputs_error = targets - final_outputs 
        # hidden layer error 
        hidden_error = np.dot(self.who.T, outputs_error)
        
        # updating the weight between hidden and output  layers 
        self.who += self.lr * np.dot((outputs_error * final_outputs * 
                                      (1 - final_outputs)), np.transpose(hidden_outputs))

        # updating the weight between hidden and input layers
        self.wih += self.lr * np.dot((hidden_error * hidden_outputs * 
                                      (1 - hidden_outputs)), np.transpose(inputs))



    # quering to data
    def query(self, inputs_list): 
        # calculating inptus in each layer of neural network 
        inputs = np.array(inputs_list, ndmin = 2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs) 
        final_inputs = np.dot(self.who, hidden_outputs) 
        final_outputs = self.activation_function(final_inputs) 

        # result of network 
        return final_outputs 

        


# neural network parameters
input_neurons = 784
hidden_neurons = 200
output_neurons = 10
learningRate = .1
#---------------------------

nn = neuralNetwork(input_neurons, hidden_neurons, output_neurons, learningRate) 

# number of training cycles
epochs = 2

# train neural network 
for e in range(epochs):
    for record in train_list:
        all_values = record.split(',') 
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_neurons) + 0.01
        targets[int(all_values[0])] = .99
        nn.train(inputs, targets) 


score = []


for record in test_list: 
    all_values = record.split(',') 
    correct_label = int(all_values[0])
    print(correct_label, "correct label") 
    inputs = (np.asfarray(all_values[1:]) / 255.0 * .99) + .01
    outputs = nn.query(inputs)
    label = np.argmax(outputs)
    print(label, "network's answer")
    if (label == correct_label): 
        score.append(1)
    else: 
        score.append(0)





print(score)

scorecard_array = np.asarray(score)
print("performance =", scorecard_array.sum()/ scorecard_array.size)
 

# to check output for specific number 
"""
all_values = test_list[index_of_specific_number].split(',')
input = np.asfarray(all_values[1:]) / 255.0 * .99
output = nn.query(input) + .01
"""

# another way to view nn result 
"""
for record in test_list: 
    all_values = record.split(',')
  
    input = np.asfarray(all_values[1:]) / 255.0 * .99

    output = nn.query(input) + .01
    label = np.argmax(output)
    print("Number: ", all_values[0])
    print("Output: ", output)
    print("Label: ", label)
"""

