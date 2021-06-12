from Neuron import Neuron


def createNetwork(layers, neuronsHidden, neuronsOutput, features):
    network = []
    for i in range(layers):
        if i + 1 == layers:  # If output layer
            outputLayer = []
            for j in range(neuronsOutput):
                outputLayer.append(Neuron(neuronsHidden))  # Add neuron to output layer

            network.append(outputLayer)  # Add output layer to network

        elif i == 0:  # If first layer
            layer = []
            # print("First layer created:")
            for j in range(neuronsHidden):
                # Adding one extra weight for bias
                layer.append(Neuron(features + 1))

            network.append(layer)  # Add First layer to network

        else:  # If layer is in the middle
            layer = []
            for j in range(neuronsHidden):
                layer.append(Neuron(neuronsHidden))

            network.append(layer)  # Add layer to network

    return network


def main():
    # The Neural Network will be a 2D Array holding Neurons
    neuralNetwork = createNetwork(2, 4, 1, 2)
    print("Neural Network created, current weights: ")
    for l in range(len(neuralNetwork)):
        for n in range(len(neuralNetwork[l])):
            print("Layer ", l + 1, ", Neuron", n + 1, neuralNetwork[l][n].weights)
    # 28 Points Dataset
    x_train = [[4, 10],
               [5, 11],
               [6, 12],
               [6, 13],
               [7, 11],
               [8, 12],
               [9, 11],
               [9, 12],
               [10, 10],
               [11, 11],
               [12, 9],
               [13, 9],
               [13, 8],
               [13, 7],
               [3, 6],
               [5, 6],
               [5, 8],
               [6, 8],
               [6, 9],
               [7, 7],
               [7, 9],
               [8, 6],
               [8, 8],
               [9, 6],
               [9, 8],
               [9, 9],
               [10, 5],
               [10, 7]]
    x_label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Training #
    # 1 - Adding extra '1' for Bias weight
    for i in range(len(x_train)):
        x_train[i].append(1)

    # 2 - Training
    epoch = 100
    for c in range(epoch):
        error_sum = 0
        for x in range(len(x_train)):  # For every point in x_train
            # print("Training: ", x_train[x])
            # 2.1 Forward Propagation
            for i in range(len(neuralNetwork)):  # In Range of Network Layers
                # Decide the input for this layer
                if i == 0:  # If first layer
                    prev_output = x_train[x]  # Input is features of the point (sample)
                else:
                    prev_output = []  # Input is outputs of previous layer
                    for ii in range(len(neuralNetwork[i - 1])):  # In Range of Neurons in previous Layer
                        prev_output.append(neuralNetwork[i - 1][ii].getOutput())

                for j in range(len(neuralNetwork[i])):  # In Range of Neurons in current Layer
                    output = neuralNetwork[i][j].calcOutput(prev_output)
                    if i + 1 == len(neuralNetwork):  # if last layer, calc error
                        error_sum += 0.5 * (x_label[x] - output) * (x_label[x] - output)

            # 2.2 Backward Propagation
            for i in range(len(neuralNetwork)):  # In Range of Network Layers
                layerJ = len(neuralNetwork) - 1 - i  # Current Layer
                eta = 0.005  # Learning rate (Given)
                if i == 0:  # Starting from last layer (Output Layer)
                    for j in range(len(neuralNetwork[layerJ])):  # In Range of Neurons in current Layer
                        # Calculating weight change
                        oj = neuralNetwork[layerJ][j].output
                        delta = (x_label[x] - oj) * (1 - oj) * oj  # Update delta in neuron
                        neuralNetwork[layerJ][j].setDelta(delta)
                        weight_changes = []
                        for k in range(len(neuralNetwork[layerJ][j].weights)):  # outputs from previous layer
                            weight_changes.append(eta * delta * neuralNetwork[layerJ - 1][k].output)
                        # Store weight till it can update
                        neuralNetwork[layerJ][j].storeWeightChanges(weight_changes)

                else:  # Other hidden layers
                    for j in range(len(neuralNetwork[layerJ])):  # In Range of Neurons in current Layer
                        # Calculating weight change
                        oj = neuralNetwork[layerJ][j].output
                        if i + 1 == len(neuralNetwork):  # If this was the first layer
                            oi = x_train[x]
                        else:
                            oi = []
                            for k in range(len(neuralNetwork[layerJ][j].weights)):
                                oi.append(neuralNetwork[layerJ - 1][k].output)
                        sumK = 0
                        for k in range(len(neuralNetwork[layerJ + 1])):  # Calculate Sum(Wkj * Delta-k)
                            sumK += (neuralNetwork[layerJ + 1][k].weights[j]) * neuralNetwork[layerJ + 1][k].delta
                            neuralNetwork[layerJ + 1][k].updateWeights()  # Allowing the neuron to update its weights

                        delta = sumK * (1 - oj) * oj
                        neuralNetwork[layerJ][j].setDelta(delta)  # Update delta in neuron
                        weight_changes = []
                        for k in range(len(neuralNetwork[layerJ][j].weights)):  # outputs from previous layer
                            weight_changes.append(eta * delta * oi[k])
                        # Store weight changes till it is allowed to update weights
                        neuralNetwork[layerJ][j].storeWeightChanges(weight_changes)

                # If this was the first layer (Last in backward propagation)
                # Allow it to update its weights
                if i + 1 == len(neuralNetwork):
                    for j in range(len(neuralNetwork[layerJ])):  # In Range of Neurons in current Layer
                        neuralNetwork[layerJ][j].updateWeights()

        print("Cycle ", c + 1, " error: ", error_sum/len(x_train))
    print("\nNew Network weights: ")
    for l in range(len(neuralNetwork)):
        for n in range(len(neuralNetwork[l])):
            print("Layer ", l + 1, ", Neuron", n + 1, neuralNetwork[l][n].weights)


if __name__ == '__main__':
    main()
