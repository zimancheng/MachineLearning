import numpy as np
import operator 
import matplotlib as plt
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.0], [1.0, 1.1], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataset, labels, k):
    """
    input: new observation, dataset that contains all training observations,
           labels that each training observation belongs to
           k: the nearest K neighbors
    output: the class that input observation belongs to
    """
    #calculate the Euclidean distance between inX and all other training data
    obsvNum = dataset.shape[0]
    diffMap = np.tile(inX, (obsvNum, 1)) - dataset
    diffSq = diffMap**2
    distSq = diffSq.sum(axis=1)
    dist = distSq**0.5
    #sort dist in ascending order for the top k points
    sortedDistIndices = dist.argsort()
    #create a dict containing all training points' classes
    labelDict = {}
    for i in range(k):
        label = labels[sortedDistIndices[i]]
        labelDict[label] = labelDict.get(label, 0) + 1
    #sort label dict in ascending order, it returns a sorted list
    labelListSorted = sorted(labelDict.items(), key = lambda x: x[1], reverse = True)
    #return the majority vote of class
    return labelListSorted[0][0]

def file2matrix(filename):
    '''
    input:  a filename string pointing to the location of the file
    output: a matrix of training examples and a list of labels for each person
    '''
    #readline and convert lines to a array
    file = open(filename)
    lines = file.readlines()
    lineNum = len(lines)

    #for each line, put the first 3 cols as features and put them in a featere matrix
    love_dict = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    ftrMat = np.zeros((lineNum, 3))
    lblArr = []
    index = 0

    for line in lines:
        line = line.strip()
        lineArr = line.split('\t')
        ftrMat[index, :] = lineArr[0:3]
        if (lineArr[-1].isdigit()):
            lblArr.append(lineArr[-1])
        else:
            lblArr.append(love_dict.get(lineArr[-1]))
        index += 1

    return ftrMat, lblArr

def autoNorm(dataSet):
    '''
    input: dataset containing all predictor data
    output: a scaled dataset with mean 0 and standard deviation 1
            range of each predictor
            minimum number of each predictor
    '''
    #calculate the min & max values of each feature
    minVal = dataSet.min(axis = 0)
    maxVal = dataSet.max(axis = 0)
    rangVal = maxVal - minVal

    #create a normalized matrix and put into all normalized feature values
    normMat = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normMat =  (dataSet - np.tile(minVal, (m, 1)))/(np.tile(rangVal, (m, 1)))
    return normMat, rangVal, minVal

def datingClassTest():
    '''
    input: none
    output: the total error rate of this KNN classifier of K = 3
    '''
    #generate test set and training set for KNN classifier
    datingMat, datingLbl = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingMat)
    m = normMat.shape[0]

    hoRatio = 0.1 #hold out ratio 10%
    testNum = int(m*hoRatio)
    errCnt = 0    

    #for each test set, count the errors
    for i in range(testNum):
        label = classify0(normMat[i, :], normMat[testNum:m, :], datingLbl[testNum:m], 3)
        print(f"the classifier came back with: {label}, the real answer is: {datingLbl[i]}")
        if label != datingLbl[i]:
            errCnt += 1
    return print(f"Test Error Rate is: {errCnt/float(testNum)}")
    
def classifyPerson():
    '''
    input: take input from user
    output: the class the person belongs to
            
    note that: love_dict = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    '''

# print("test for reloading module")

