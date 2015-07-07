
import sys,os,gzip,pickle
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score

from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone

from RunExperiments import ML, ResultPrinter

# the newer version which do not use all labels in training data

libFolder = '/home/r02922010/master_thesis/dimReduction/JGibbLabeledLDA'
modelFolder='%s/model_folder' % (libFolder)
dataFolder = '/home/r02922010/master_thesis/data'


# y: original y
# trainIndex: original training data index 
# testIndex: original testing data index (the docs id to ignore)
def gridSearch(taskName, dataFile, topic, nTopics, nIters, alpha, beta, y, trainIndex, testIndex, clfName, n_folds=10, n_jobs=-1):
    kfold = StratifiedKFold(y[trainIndex], n_folds=n_folds)
    # valDocIds[i]: the list of document id in testing set of validation stage
    testDocIds = list()
    for valTrainIndex, valTestIndex in kfold:
        docIds = [trainIndex[i] for i in valTestIndex]
        docIds.extend(testIndex)
        testDocIds.append(docIds)

    # dimension reduction for each fold
    allThetaList = Parallel(n_jobs=n_jobs)(
                delayed(runLLDA)('%s_vfold%d' % (taskName, fi), dataFile, topic, nTopics, nIters, alpha, beta, testDocIds[fi], readTopicWord=False) 
            for fi in range(0, n_folds))
    thetaList = [None for i in range(0, len(allThetaList))]
    for i in range(0, len(allThetaList)):
        thetaList[i] = allThetaList[i][trainIndex]
    
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
    bestGridStart = None
    n_fits = len(out)
    for grid_start in range(0, n_fits, n_folds):
        avgScore = 0.0
        for r in out[grid_start:grid_start + n_folds]:
            avgScore += r['test']
        avgScore /= n_folds
        if bestScore is None or avgScore > bestScore:
            bestScore = avgScore
            bestParams = out[grid_start]['params']
            bestGridStart = grid_start

    allClf = list()
    for r in out[bestGridStart:bestGridStart + n_folds]:
        allClf.append(r['clf'])

    return bestParams, bestScore, allThetaList, allClf  #best parameters, and best validation score


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
            'yTestPredict': yTestPredict, 'params': params, 'clf': clf }

def runLLDA(taskName, dataFile, topic, nTopics, nIters, alpha, beta, ignoreDocIds, readTopicWord=False):
    dataFile = '%s.gz' % (dataFile)
    modelFile = '%s/%s' % (modelFolder, taskName)
    ignoreDocIdsStr = genIdStr(ignoreDocIds)
    osRunLLDA(dataFile, nTopics, nIters, alpha, beta, ignoreDocIdsStr, modelFile)
    theta = readTheta('%s.theta.gz' % (modelFile))
    if readTopicWord:
        topicWords = readTopicWord('%s.twords.gz' % (modelFile), nTopics)
        removeFile(taskName)
        return theta, topicWords
    else:
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
def osRunLLDA(dataFile, nTopics, nIters, alpha, beta, ignoreDocIdsStr, modelFile):
    cmd = 'cd %s; make run -s dir="/" dataFile="%s" nTopics="%d" nIters="%d" alpha="%f" beta="%f" modelName="%s" igDocIds="%s"' %(
            libFolder, dataFile, nTopics, nIters, alpha, beta, modelFile, ignoreDocIdsStr)
    #print(cmd)
    os.system(cmd)

def removeFile(taskName):
    cmd = 'rm %s/%s.*.gz' % (modelFolder, taskName)
    os.system(cmd)

# read theta(doc-topic distribution) from gzip file (from JGibbsLLDA)
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
 
# read top words in each topic
def readTopicWord(gzFile, nTopics):
    topicWords = [list() for i in range(0, nTopics)]
    nowTopic = 0
    with gzip.open(gzFile, 'rt') as f:
        for line in f:
            if line.find('Topic') != -1:
                nowTopic = int(line[line.find('Topic ') + len('Topic '):line.find(':')])
            else:
                entry = line.strip().split()
                topicWords[nowTopic].append(entry[0])
    return topicWords


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
    if len(sys.argv) < 9:
        print('Usage:', sys.argv[0], 'Topic DataFile(abs_path) RandSeed nTopics nIters alpha beta taskName [outFilePrefix]', file=sys.stderr)
        exit(-1)

    topic = int(sys.argv[1])
    dataFile = sys.argv[2]
    randSeed = int(sys.argv[3])
    nTopics = int(sys.argv[4])
    nIters = int(sys.argv[5])
    alpha = float(sys.argv[6])
    beta = float(sys.argv[7])
    taskName = sys.argv[8]
    outFilePrefix = sys.argv[9] if len(sys.argv) == 10 else None

    labels, labelIndex, noLabelIndex = readLLDADataLabel(dataFile)
    print('#all doc:', len(labels), ' #labeled:', len(labelIndex), ' #unlabeled:', len(noLabelIndex), file=sys.stderr)
    clf = ML.genClf( {'clfName':'MaxEnt', 'params': {}} )

    # run training and testing process using cross-validation
    ResultPrinter.printFirstLine()
    y = labels[labelIndex]
    kfold = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=randSeed)
    for fid, (trainIndex, testIndex) in enumerate(kfold):
        (bestParam, bestValScore, allThetaList, allClf) = gridSearch(taskName, dataFile, topic, nTopics, nIters, alpha, beta, y, trainIndex, testIndex, 'MaxEnt', n_folds=10, n_jobs=-1)
        
        assert len(allThetaList) == len(allClf)
        allP = list()
        for i in range(0, len(allClf)):
            p = allClf[i].predict(allThetaList[i][testIndex])
            allP.append(p)
        allP = np.array(allP)
        avgP = np.sum(allP, axis=0)
        for i in avgP:
            if avgP[i] > 5:
                avgP[i] = 1
            else:
                avgP[i] = 0

        print(accuracy_score(y[testIndex], avgP))

        theta, topicWords = runLLDA('%s_fold%d' % (taskName, fid), dataFile, topic, nTopics, nIters, alpha, beta, testIndex, readTopicWord)
        r = oneTask(theta, y, trainIndex, testIndex, clone(clf), bestParam)
        print(r)
        ResultPrinter.print('SelfTrainTest', 'MaxEnt', 'Accurarcy', theta.shape[1], randSeed, fid, r['train'], bestValScore, r['test']) 
        
        if outFilePrefix is not None:
            p = { 'X': theta, 'y': y, 'topicWords': topicWords, 'split': { 'trainIndex': trainIndex, 'testIndex': testIndex } }
            with open(outFilePrefix + '_fold%d.pickle' % (fid), 'w+b') as f:
                pickle.dump(p, f)


