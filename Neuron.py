import random
import math


class Neuron:
    # Initialize A Neuron with random weights
    def __init__(self, weightsNum):
        self.weights = []  # Wj
        self.output = 0  # Oj
        self.delta = 0  # Delta-j
        # Holding the weight changes until the
        # previous layer takes the old weights (Wk)
        self.weightChanges = []
        for i in range(weightsNum):
            self.weights.append(random.randint(-10, 10))

    def getWeights(self):
        return self.weights

    def getOutput(self):
        return self.output

    def setDelta(self,new_delta):
        self.delta=new_delta

    def calcOutput(self, inputs):  # Calculate Neuron Output (Forward Propagation)
        net = 0  # Net of the Neuron
        for i in range(len(inputs)):
            net += inputs[i] * self.weights[i]
        # Assign Neuron output + Activation
        self.output = self.actFun(net)
        return self.output

    def actFun(self, net):  # Sigmoid Activation Function
        expo = math.exp(-1 * net)
        activated = 1 / (1 + expo)
        return activated

    def storeWeightChanges(self, changes):  # Store Weight Changes
        self.weightChanges = changes

    def updateWeights(self):  # Updates weights
        # wNew = wOld + Change in Weight
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.weightChanges[i]
