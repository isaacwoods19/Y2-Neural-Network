import csv
from random import randrange, uniform
from math import exp

#This collects the data from the csv file and puts it in an array
Data = []
with open('CleanedData.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            Data.append(row)
            line_count += 1

#changes relevant fields from string to float values so we can later use them for calculations
for j in range (0,len(Data)):
    for i in range(1,7):
        Data[j][i] = float(Data[j][i])
    #remove the date part of the data
    Data[j].pop(0)



#finding the max and min of each field to be used in standardisation
maxT = max([_[0] for _ in Data])
minT = min([_[0] for _ in Data])
maxW = max([_[1] for _ in Data])
minW = min([_[1] for _ in Data])
maxSR = max([_[2] for _ in Data])
minSR = min([_[2] for _ in Data])
maxDSP = max([_[3] for _ in Data])
minDSP = min([_[3] for _ in Data])
maxDRH = max([_[4] for _ in Data])
minDRH = min([_[4] for _ in Data])
maxP = max([_[5] for _ in Data])
minP = min([_[5] for _ in Data])

#standardisation formula is now applied to the data
for i in range (0,len(Data)):
    Data[i][0] = 0.8 * ((Data[i][0] - minT) / (maxT - minT)) + 0.1
    Data[i][1] = 0.8 * ((Data[i][1] - minW) / (maxW - minW)) + 0.1
    Data[i][2] = 0.8 * ((Data[i][2] - minSR) / (maxSR - minSR)) + 0.1
    Data[i][3] = 0.8 * ((Data[i][3] - minDSP) / (maxDSP - minDSP)) + 0.1
    Data[i][4] = 0.8 * ((Data[i][4] - minDRH) / (maxDRH - minDRH)) + 0.1
    Data[i][5] = 0.8 * ((Data[i][5] - minP) / (maxP - minP)) + 0.1


#Splitting the data up into three sets, training, validation and testing
indexes = []

TrainingData = []
#876.6 is 60% of the 1461 records
for i in range (0,877):
    #simply randomly generates an index from Data, adds it to Training, then removes it from Data so it cant be reselected
    random_index = randrange(len(Data))
    TrainingData.append(Data[random_index])
    del Data[random_index]

ValidationData = []
#292.2 is 20% of 1461
for i in range (0,292):
    random_index = randrange(len(Data))
    ValidationData.append(Data[random_index])
    del Data[random_index]

TestingData = []
#there is only 292 records left
for i in range(0, 292):
    random_index = randrange(len(Data))
    TestingData.append(Data[random_index])
    del Data[random_index]



###MAIN ALGORITHM###

#Forward pass function
def ForwardPass(Data, HiddenLayer, OutputLayer, point):
    #for each hidden node
    for i in range(0, len(HiddenLayer)):
        Sum = 0
        #add up all weights multiplied by their connected input
        for j in range(0, len(Data[point]) - 1):
            Sum += HiddenLayer[i][j+1] * Data[point][j]

        #add the bias at the end
        Sum += HiddenLayer[i][0]
        #simple sigmoid function, then stored in the OutputLayer array for later
        OutputLayer[i+2][1] = 1/(1 + exp(-Sum))

    #calculating output
    SO = 0
    #take all of the hidden node U values and multiply them by their weights
    for i in range(2, len(OutputLayer)):
        SO += OutputLayer[i][1] * OutputLayer[i][0]

    #add the output bias at the end
    SO += OutputLayer[1][0]
    #apply the sigmoid function
    UO = 1/( 1 + exp(-SO) )
    #Output#
    OutputLayer[0] = UO
        

def ANN(NumNodes, LearningParameter, TotalEpochs):

    #HiddenLayer[x][y] takes the weight on arc between input node x and hidden node y
    #HiddenLayer[x][0] is the bias for hidden node x
    HiddenLayer = []
    #OutputLayer[0] is the final output of the network
    #OutputLayer[1] is the bias
    #other than those, OutputLayer[x][0] is the Weight between hidden node x and the output node
    #OutputLayer[x][1] is all of the hidden nodes' U values
    OutputLayer = [0, [1, 1]]
    #an array to hold the delta values
    Delta = [0]
    #ValidationRMSE[0] is the previous RMSE on the validation set, and [1] is the current RMSE
    ValidationRMSE = [0, 0]

    #populates the data structures with the correct number of sub-arrays to represent the chosen hidden nodes
    for i in range(0, NumNodes):
        HiddenLayer.append([1, 1, 1, 1, 1, 1])
        OutputLayer.append([1, 0])
        Delta.append(0)

    #Weight Randomisation#
    for i in range(0, len(HiddenLayer)):
        for j in range(0, len(HiddenLayer[0])):
            #randomly selects a real weight between (-2/n, 2/n) where n is the number of inputs
            n = len(TrainingData[0]) - 1
            HiddenLayer[i][j] = uniform(-2/n, 2/n)

    for i in range(1, len(OutputLayer)):
        m = len(HiddenLayer)
        OutputLayer[i][0] = uniform(-2/m, 2/m)

    ##MAIN EPOCH LOOP##
    for epochs in range(1, TotalEpochs+1):
        MSE = 0
        #validation set test#
        if epochs % 500 == 0:
            # every 500 epochs, checks against the validation set#
            ValidationRMSE[0] = ValidationRMSE[1]
            ValidationRMSE[1] = 0
            Difference = 0
            for point in range(0, len(ValidationData)):
                #perform a forward pass for every data point in the validation set
                ForwardPass(TrainingData, HiddenLayer, OutputLayer, point)
                c = ValidationData[point][5]

                ValidationRMSE[1] += ((c-OutputLayer[0]))**2

                
                
            ValidationRMSE[1] = (ValidationRMSE[1] / len(ValidationData))**0.5
            #gate to stop it comparing the first accuracy check with nothing
            if ValidationRMSE[0] != 0:
                Difference = ValidationRMSE[0] - ValidationRMSE[1]
                if ValidationRMSE[0] - ValidationRMSE[1] < 0:
                    #if the RMSE for the validation set starts to get worse, then it stops cycling through the epochs prematurely to stop overfitting
                    break
            
            
        ##MAIN ALGORITHM##
        for point in range(0, len(TrainingData)):
            #Forward pass#
            ForwardPass(TrainingData, HiddenLayer, OutputLayer, point)
            
            #Correct value#
            c = TrainingData[point][5]

            #Mean Squared Error#
            MSE += ((c-OutputLayer[0]))**2

            ###BACKWARD PASS###

            ##DELTA VALUES##
            end = len(Delta)-1
            #first delta node
            Fs5 = OutputLayer[0]*(1 - OutputLayer[0])
            #populate the delta array from back to front to make access easier later
            Delta[end] = (c - OutputLayer[0])*Fs5

            #rest of the nodes
            for i in range (end-1,-1,-1):
                Delta[i] = OutputLayer[i+2][0] * Delta[end] * (OutputLayer[i+2][1] * (1 - OutputLayer[i+2][1]))


            ##WEIGHT UPDATE##

            #HiddenLayer update
            for i in range(0, len(HiddenLayer)):
                for j in range(0, len(HiddenLayer[0])):
                    #if its the bias
                    if j == 0:
                        U = 1
                    #if its any other input value
                    else:
                        U = TrainingData[point][j-1]

                    #weight update formula
                    HiddenLayer[i][j] = HiddenLayer[i][j] + (LearningParameter * Delta[i] * U)

            #OutputLayer update - same as above
            for i in range(1, len(OutputLayer)):
                OutputLayer[i][0] = OutputLayer[i][0] + (LearningParameter * Delta[-1] * OutputLayer[i][1])

        #error value calculation
        MSE = MSE/len(TrainingData)
        RMSE = (MSE)**0.5

    #prints to console how many epochs it reached before being aborted (stop overfitting)
    #and shows all error values it reached
    print(epochs, "Epochs")
    print("MSE =", MSE)
    print("RMSE =", RMSE)
    print("Validation RMSE =", ValidationRMSE[1])


    ##FINAL ERROR TEST##
    FinalRMSE = 0
    #for each data point in the test data
    for point in range(0, len(TestingData)):
        #forward pass
        ForwardPass(TestingData, HiddenLayer, OutputLayer, point)
        c = TestingData[point][5]

        FinalRMSE += (c - OutputLayer[0]) ** 2
    FinalRMSE = (FinalRMSE/len(TestingData))**0.5
    print("Final accuracy", FinalRMSE)

#Function call to start it all up#
print("5 nodes, 0.4")
ANN(5, 0.4, 10000)
