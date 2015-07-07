
import sys, pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
import climate
import theanets

from scipy.io import mmwrite, mmread

from misc import *
from Volc import Volc

def readSVMData(filename, maxDim):
    ydata = list()
    Xdata = list()
    with open(filename, 'r') as f:
        for line in f:
            entry = line.strip().split(' ') 
            ydata.append(int(entry[0]))
            d = dict()
            for e in entry[1:]:
                w,v = e.strip().split(':')
                w = int(w)
                v = float(v)
                d[w] = v
                assert v <= maxDim
            Xdata.append(d)
    # generating csr matrix
    rows = list()   
    cols = list()
    data = list()
    for i, x in enumerate(Xdata):
        for w, v in x.items():
            rows.append(i)
            cols.append(w - 1)
            data.append(v)
    X = csr_matrix((data, (rows, cols)), shape=(len(Xdata), maxDim), dtype=np.float64)
    y = np.array(ydata)
    return X, y


def divideUnlabeledData(X, y):
    topicIndex = dict()
    for i, yi in enumerate(y):
        if yi not in topicIndex:
            topicIndex[yi] = list()
        topicIndex[yi].append(i)
    topicX = dict()
    for t, indexList in topicIndex.items():
        topicX[t] = X[indexList]
    return topicX, sorted(topicIndex.keys())

def divideLabeledData(X, y):
    # original y is not real label
    topicIndex = dict()
    labels = list()
    for i, yi in enumerate(y):
        label = 0 if yi < 0 else 1
        labels.append(label)
        t = abs(yi)
        if t not in topicIndex:
            topicIndex[t] = list()
        topicIndex[t].append(i)
    y = np.array(labels, dtype=np.uint8)
    #y = y.reshape((len(y), -1))
    topicX = dict()
    topicy = dict()
    for t, indexList in topicIndex.items():
        topicX[t] = X[indexList]
        topicy[t] = y[indexList]
    return topicX, topicy, sorted(topicIndex.keys())

# for the data in each topic, reduce the dimension
def prepareData(topiclX, topicUnX, topicList, minCnt=4):
    newTopiclX = dict()
    topicAllX = dict()
    for t in topicList:
        lX, unX = topiclX[t], topicUnX[t]
        labelNum, unLabelNum = lX.shape[0], unX.shape[0]
        allX = vstack((lX, unX))
        DF = countDFByCSRMatrix(allX)
        allX = shrinkCSRMatrixByDF(allX, DF, minCnt=minCnt)
        lX = allX[0:labelNum]
        newTopiclX[t] = lX
        topicAllX[t] = allX
    return (topiclX, topicAllX)


def splitTrainValTest(X, y, train, val):
    assert train + val < 1.0
    assert X.shape[0] == len(y)
    num = len(y)
    trainNum = int(num * train)
    valNum = int(num * val)
    testNum = num - trainNum - valNum
    trainIndex = range(0, trainNum)
    valIndex = range(trainNum, trainNum + valNum)
    testIndex = range(trainNum + valNum, trainNum + valNum + testNum)
    return (X[trainIndex], y[trainIndex]), (X[valIndex], y[valIndex]), (X[testIndex], y[testIndex])


def readData(filename):
    if filename.rfind('.mtx') != -1:
        return mmread(filename).toarray()
    elif filename.find('.npy') != -1:
        return np.load(filename)
    
def readPickle(filename):
    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)
    
    if type(p['X']) == csr_matrix:
        X = p['X'].todense()
    else:
        X = p['X']
    y = p['y'].astype(np.uint8)
    print(type(y), y.shape)
    return X, y

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:', sys.argv[0], 'data(.pickle) middleLayer [pretrainModel]', file=sys.stderr)
        exit(-1)
 
    climate.enable_default_logging()
    pickleFile = sys.argv[1]
    middleLayer = int(sys.argv[2])
    modelName = None
    if len(sys.argv) == 4:
        modelName = sys.argv[3]

    X, y = readPickle(pickleFile)
    print(X.shape, X.dtype, type(X))
    print(y.shape, y.dtype, type(y))
    print('labelData: ', X.shape, file=sys.stderr)

    # cross validation
    kfold = StratifiedKFold(y, n_folds=10)
    for trainIndex, testIndex in kfold:
        XTrain, yTrain = X[trainIndex], y[trainIndex]
        XTest, yTest = X[testIndex], y[testIndex]
        
        exp = theanets.Experiment( 
            theanets.Classifier,
            layers=(X.shape[1], middleLayer, 2),
        )

        if modelName is not None:
            exp.network.load_params(modelName)

        exp.train(
            (XTrain, yTrain),
            optimize='rmsprop',
            learning_rate=0.001,
            momentum=0.5,
            hidden_l1=0.001,
            hidden_dropout=0.5
        )
        
        predict = exp.network.classify(XTest)
        print(accuracy_score(yTest, predict))
        
