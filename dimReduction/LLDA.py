
import sys,os,gzip
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score

from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone

from RunExperiments import ML, ResultPrinter

libFolder = '/home/r02922010/master_thesis/dimReduction/JGibbLabeledLDA'
modelFolder='%s/model_folder' % (libFolder)
dataFolder = '/home/r02922010/master_thesis/data'


# y: original y
# trainIndex: original training data index 
# testIndex: original testing data index (the docs id to ignore)
def gridSearch(taskName, dataFile, topic, nTopics, nIters, y, trainIndex, testIndex, clfName, n_folds=10, n_jobs=-1):
    kfold = StratifiedKFold(y[trainIndex], n_folds=n_folds)
    # valDocIds[i]: the list of document id in testing set of validation stage
    testDocIds = list()
    for valTrainIndex, valTestIndex in kfold:
        docIds = [trainIndex[i] for i in valTestIndex]
        docIds.extend(testIndex)
        testDocIds.append(docIds)

    # dimension reduction for each fold
    thetaList = Parallel(n_jobs=n_jobs)(
                delayed(runLLDA)('%s_vfold%d' % (taskName, fi), dataFile, topic, nTopics, nIters, testDocIds[fi]) 
            for fi in range(0, n_folds))
    for i in range(0, len(thetaList)):
        thetaList[i] = thetaList[i][trainIndex]
    
    # create clf & params
    (clf, paramDict) = ML.genClfAndParams(clfName)
    paramsGrid = ParameterGrid(paramDict)

    # doing grid search 
    yTrain = y[trainIndex]
    out = Parallel(n_jobs=n_jobs)(delayed(oneTask)(thetaList[fi], yTrain, valTrainIndex, valTestIndex, clone(clf), params) 
                for params in paramsGrid 
                for fi, (valTrainIndex, valTestIndex) in enumerate(kfold))
    
    # collecting results to find best parameters
    bestScore = None
    n_fits = len(out)
    for grid_start in range(0, n_fits, n_folds):
        avgScore = 0.0
        for r in out[grid_start:grid_start + n_folds]:
            avgScore += r['test']
        avgScore /= n_folds
        if bestScore is None or avgScore > bestScore:
            bestScore = avgScore
            bestParams = out[grid_start]['params']
    return bestParams, bestScore  #best parameters, and best validation score


def oneTask(X, y, trainIndex, testIndex, clf, params):
    XTrain, yTrain = X[trainIndex], y[trainIndex]
    XTest, yTest = X[testIndex], y[testIndex]
    clf.set_params(**params)
    clf.fit(XTrain, yTrain)
    yTrainPredict = clf.predict(XTrain)
    yTestPredict = clf.predict(XTest)
    train = accuracy_score(yTrain, yTrainPredict)
    test = accuracy_score(yTest, yTestPredict)
    return { 'test': test, 'train': train, 'yTrainPredict': yTrainPredict, 
            'yTestPredict': yTestPredict, 'params': params }

def runLLDA(taskName, dataFile, topic, nTopics, nIters, ignoreDocIds):
    dataFile = '%s.gz' % (dataFile)
    modelFile = '%s/%s' % (modelFolder, taskName)
    ignoreDocIdsStr = genIdStr(ignoreDocIds)
    osRunLLDA(dataFile, nTopics, nIters, ignoreDocIdsStr, modelFile)
    theta = readTheta('%s.theta.gz' % (modelFile))
    removeFile(taskName)
    return theta

def genIdStr(idList):
    outStr = ''
    for id in idList:
        if len(outStr) == 0:
            outStr = '%d' % (id)
        else:
            outStr += ' %d' % (id)
    return outStr

# call command
def osRunLLDA(dataFile, nTopics, nIters, ignoreDocIdsStr, modelFile):
    cmd = 'cd %s; make run -s dir="/" dataFile="%s" nTopics="%d" nIters="%d" modelName="%s" igDocIds="%s"' %(
            libFolder, dataFile, nTopics, nIters, modelFile, ignoreDocIdsStr)
    #print(cmd)
    os.system(cmd)

def removeFile(taskName):
    cmd = 'rm %s/%s.*' % (modelFolder, taskName)
    os.system(cmd)

# read theta from gzip file (from JGibbsLLDA)
def readTheta(gzFile):
    theta = list()
    with gzip.open(gzFile, 'rt') as f:
        for line in f:
            entry = line.strip().split(' ')
            dist = list()
            for e in entry:
                (index, p) = e.split(':')
                dist.append(float(p))
            theta.append(dist)
    return np.array(theta)
 

def readLLDADataLabel(filename):
    labels = list()
    labelIndex = list()
    noLabelIndex = list()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            s = line.find('[')
            e = line.find(']')
            if s == -1 or e == -1: # no label
                labels.append(-1)
                noLabelIndex.append(i)
            else:
                labels.append(int(line[s+1:e]))
                labelIndex.append(i)
    return np.array(labels), labelIndex, noLabelIndex




if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('Usage:', sys.argv[0], 'Topic DataFile(abs_path) RandSeed nTopics nIters taskName', file=sys.stderr)
        exit(-1)

    topic = int(sys.argv[1])
    dataFile = sys.argv[2]
    randSeed = int(sys.argv[3])
    nTopics = int(sys.argv[4])
    nIters = int(sys.argv[5])
    taskName = sys.argv[6]

    labels, labelIndex, noLabelIndex = readLLDADataLabel(dataFile)
    print('#all doc:', len(labels), ' #labeled:', len(labelIndex), ' #unlabeled:', len(noLabelIndex), file=sys.stderr)
    clf = ML.genClf( {'clfName':'MaxEnt', 'params': {}} )

    # run training and testing process using cross-validation
    ResultPrinter.printFirstLine()
    y = labels[labelIndex]
    kfold = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=randSeed)
    for fid, (trainIndex, testIndex) in enumerate(kfold):
        (bestParam, bestValScore) = gridSearch(taskName, dataFile, topic, nTopics, nIters, y, trainIndex, testIndex, 'MaxEnt', n_folds=10, n_jobs=-1)
        
        theta = runLLDA('t%d_fold%d' % (topic, fid), dataFile, topic, nTopics, nIters, testIndex)
        r = oneTask(theta, y, trainIndex, testIndex, clone(clf), bestParam)
        ResultPrinter.print('SelfTrainTest', 'MaxEnt', 'Accurarcy', theta.shape[1], randSeed, fid, r['train'], bestValScore, r['test']) 

