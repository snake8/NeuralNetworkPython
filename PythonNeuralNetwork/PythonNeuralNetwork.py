"""
FULL DATASET: https://pjreddie.com/projects/mnist-in-csv/
"""
import scipy.special
import numpy as np
import matplotlib.pyplot as plt

def creatingDataListFromDataFile(dataFile):
    dataList = dataFile.readlines()
    dataFile.close()
    return dataList

trainingDataFile = open("datafolder/mnist_train_100.csv")
testDataFile = open("datafolder/mnist_test_10.csv")
trainDataList = creatingDataListFromDataFile(trainingDataFile)
testDataList = creatingDataListFromDataFile(testDataFile)

class neuralNetwork(object):
    def __init__(self, inputNeurons, hiddenNeurons, outputNeurons, learningRate):
        self.inputsNeurons = inputNeurons
        self.hiddenNeruons = hiddenNeurons
        self.outputNEurons = outputNeurons
        self.learningRate = learningRate
        #sigmoid function implementation in python
        self.activation_function = lambda x: scipy.special.expit(x)
        self.weightFromInputToHidden = np.random.normal(0.0, pow(hiddenNeurons, -0.5),
                                    (hiddenNeurons, inputNeurons))
        self.weightFromHiddenToOutput = np.random.normal(0.0, pow(outputNeurons, 0.5),
                                    (outputNeurons, hiddenNeurons))

    def trainingNetwork(self, inputList, targetList):
        inputs = np.array(inputList, ndmin = 2).T
        targetList = np.array(targetList, ndmin = 2).T

        hiddenNodes = np.dot(self.weightFromInputToHidden, inputs)
        hiddenOutputs = self.activation_function(hiddenNodes)
        finalInputs = np.dot(self.weightFromHiddenToOutput, hiddenOutputs)
        finalOutputs = self.activation_function(finalInputs)

        outputError = targetList - finalOutputs
        hiddenError = np.dot(self.weightFromHiddenToOutput.T, outputError)

        self.weightFromHiddenToOutput += learningRate * np.dot((outputError * finalOutputs *
                                    (1 - finalOutputs)), np.transpose(hiddenOutputs))
        self.weightFromInputToHidden += learningRate * np.dot((hiddenError * hiddenOutputs *
                                    (1 - hiddenOutputs)), np.transpose(inputs))


        print('Training in process!')

    def queryNetwork(self, inputs_list):
        inputs = np.array(inputs_list, ndmin = 2).T
        hiddenInputs = np.dot(self.weightFromInputToHidden, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)
        finalInputs = np.dot(self.weightFromHiddenToOutput, hiddenOutputs)
        finalOutputs = self.activation_function(finalInputs)
        return finalOutputs

inputNeurons = 784
hiddenNeurons = 200
outputNeurons = 10
learningRate = .1

nn = neuralNetwork(inputNeurons, hiddenNeurons, outputNeurons, learningRate)

trainingCycles = 100
# training neural network
for e in range(trainingCycles):
    for record in trainDataList:
        allValues = record.split(',')
        inputInNetwork = (np.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(outputNeurons) + 0.01
        targets[int(allValues[0])] = .99
        nn.trainingNetwork(inputInNetwork, targets)
        print('Training in process!')

score = []

for record in testDataList:
    allValues = record.split(',')
    correctLabel = int(allValues[0])
    print(correctLabel, "correct label")
    inputValues = (np.asfarray(allValues[1:]) / 255.0 * .99) + .01
    outputValues = nn.queryNetwork(inputValues)
    labelFromNetwork = np.argmax(outputValues)
    if (labelFromNetwork == correctLabel):
        score.append(1)
    else:
       score.append(0)

print(score)

scoreCardArray = np.asarray(score)
print("Performance =", scoreCardArray.sum() / scoreCardArray.size())


# to check output for specific number
'''
indexOfSpecificNumber = 0
allValues = testDataList[indexOfSpecificNumber].split(',')
inputList = np.asfarray(allValues[1:]) / 255.0 * .99
outputList = nn.queryNetwork(inputList) + .01
print(outputList)
'''


# another way to view NeuralNetwork result
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
