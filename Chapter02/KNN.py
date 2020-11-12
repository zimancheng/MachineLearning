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
    fr = open(filename)
    lines = fr.readlines()
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
            lblArr.append(int(lineArr[-1]))
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
    lblList = ['not at all', 'in small doses', 'in large doses']
    icecream = float(input("Liters of ice cream consumed per year?"))
    flyMiles = float(input("Frequent flier miles earned per year?"))
    vidPct = float(input("Percentage of time spent playing video games?"))
    inArr = np.array([flyMiles, vidPct, icecream]) #create the array as inputX

    #create the training data set
    trainMat, trainLbl = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(trainMat)

    #classify the person
    label = classify0((inArr - minVals)/ranges, normMat, trainLbl, 3)
    print(f"You will probably like this person: {lblList[label - 1]}")

def img2Vec(filename):
    '''
    input: the filedir of each img txt file
    output: the 1*1024 numpy matrix, make it matrix for future classificationn
    '''
    arr = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            arr[0, i*32 + j] = line[j]
    return arr

def handwritingClassTest():
    '''
    returns the error rate of testDigits image classification
    '''
    # set up the training data set
    # create a m*1024 matrix to contain all training image vectors
    # create label list for the true numbers of each image
    train_img_list = listdir("trainingDigits") #['7_10.txt', '7_11.txt', ...]
    train_m = len(train_img_list)
    trainMat = np.zeros((train_m, 1024))
    train_lbl_list = []

    for i in range(train_m):
        img_file_name = train_img_list[i] #'7_10.txt'
        img_lbl = int(img_file_name.split('_')[0])
        train_lbl_list.append(img_lbl)
        trainMat[i, :] = img2Vec("trainingDigits/" + img_file_name)
    
    # classify the test data set 
    test_img_list = listdir("testDigits")
    test_m = len(test_img_list)
    test_err_cnt = 0

    for i in range(test_m):
        img_file_name = test_img_list[i]
        img_lbl = int(img_file_name.split('_')[0])
        test_arr = img2Vec("testDigits/" + img_file_name)
        pred_lbl = classify0(test_arr, trainMat, train_lbl_list, 3)
        print(f"The classifier came back with: {pred_lbl}, the real answer is: {img_lbl}")
        if pred_lbl != img_lbl:
            test_err_cnt += 1
    
    print(f"\nThe total number of errors is: {test_err_cnt}")
    print(f"\nThe total error rate is: {test_err_cnt/test_m}")




# print("test for reloading module")

