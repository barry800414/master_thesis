#!/usr/bin/env python3
import sys, os, pickle
from collections import defaultdict

from sklearn.cross_validation import StratifiedKFold
from misc import *
from Volc import *

# print data for sle struct svm tool
def printDataFile(filename, docy, sentPX, docPX, sentSX, doc2XList, indexList, outfile=sys.stdout):
    assert sentPX.shape[0] == sentSX.shape[0]
    with open(filename, 'w') as outfile:
        for di in indexList:
            nSent = len(doc2XList[di])
            y = 1 if docy[di] == 1 else -1
            print(y, nSent, sep=' ', file=outfile)
            for si in range(0, nSent):
                print(si, end='', file=outfile)
                # polarity feature
                xIndex = doc2XList[di][si]
                printRow(sentPX, xIndex, outfile=outfile)
                # subjective feature
                printRow(sentSX, xIndex, prefix='S', outfile=outfile)
                print('', file=outfile)

            # document feature
            print(nSent, end='',file=outfile)
            printRow(docPX, di, outfile=outfile)
            print('\n', file=outfile)

def printRow(X, rowIndex, prefix='', outfile=sys.stdout):
    if type(X) == csr_matrix:
        row = X.getrow(rowIndex)
        (colIndex, rowPtr, data) = row.indices, row.indptr, row.data
        ivList = list()
        nowPos = 0
        for ci in colIndex[rowPtr[0]:rowPtr[1]]:
            ivList.append((ci, data[nowPos]))
            nowPos += 1
        for i, v in sorted(ivList, key=lambda x:x[0]):
            print(' %s%d:%f' % (prefix, i+1, v), end='', file=outfile)

    elif type(X) == np.ndarray:
        row = X[rowIndex]
        for i, v in sorted(enumerate(row), key=lambda x:x[0]):
            print(' %s%d:%f' % (prefix, i+1, v), end='', file=outfile)

def getInitGuess(sentSX, doc2XList, nLabelDoc, percent):
    ones = np.ones((sentSX.shape[1], 1))
    guessSentList = list()
    for di in range(0, nLabelDoc):
        sentScore = list()
        nSent = len(doc2XList[di])
        for si in range(0, nSent):
            xIndex = doc2XList[di][si]
            sentScore.append((si, sentSX.getrow(xIndex).todense() * ones))
        sentScore.sort(key=lambda x:x[1], reverse=True)
        sentList = [si for si, score in sentScore[0: round(percent *nSent)]]
        sentList.sort()
        guessSentList.append(sentList)
    return guessSentList

def printInitGuessFile(filename, sentIndexList, trainIndex):
    with open(filename, 'w') as outfile:
        for i in trainIndex:
            sentIndex = sentIndexList[i]
            print(len(sentIndex), end='', file=outfile)
            for j, si in enumerate(sentIndex):
                print(' ', si, sep='', end='', file=outfile)
            print('', file=outfile)

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage:', sys.argv[0], 'PickleFile Topic Seed GuessPercent outFolder', file=sys.stderr)
        exit(-1)

    pickleFile = sys.argv[1]
    topic = int(sys.argv[2])
    seed = int(sys.argv[3])
    percent = float(sys.argv[4])
    outFolder = sys.argv[5]

    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)

    sentPX, docPX, sentSX, y, doc2XList = p['sentPX'], p['docPX'], p['sentSX'], p['docy'], p['doc2XList']
    nLabelDoc = len(y)
    guessSentList = getInitGuess(sentSX, doc2XList, nLabelDoc, percent)

    kfold = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=seed)
    for i, (trainIndex, testIndex) in enumerate(kfold):
        print('Generating data for fold %d ...' % i, file=sys.stderr)
        prefix = 'T%dS%dF%d' % (topic, seed, i)
        os.system('mkdir -p %s/%s' % (outFolder, prefix))

        # generate validation data
        yTrain = y[trainIndex]
        valKfold = StratifiedKFold(yTrain, n_folds=10, shuffle=True, random_state=seed)
        for j, (valTrainIndex, valTestIndex) in enumerate(valKfold):
            realTrainIndex = [trainIndex[i] for i in valTrainIndex]
            realTestIndex = [trainIndex[i] for i in valTestIndex]
            prefix2 = '%s/%s/%sV%d' % (outFolder, prefix, prefix, j)
            printDataFile('%s.train' % (prefix2), y, sentPX, docPX, sentSX, doc2XList, realTrainIndex)
            printDataFile('%s.test' % (prefix2), y, sentPX, docPX, sentSX, doc2XList, realTestIndex)
            printInitGuessFile('%s.init' % (prefix2), guessSentList, realTrainIndex)
        
        # generate training and testing data
        prefix2 = '%s/%s/%s' % (outFolder, prefix, prefix)
        printDataFile('%s.train' % (prefix2), y, sentPX, docPX, sentSX, doc2XList, trainIndex)
        printDataFile('%s.test' % (prefix2), y, sentPX, docPX, sentSX, doc2XList, testIndex)
        printInitGuessFile('%s.init' % (prefix2), guessSentList, trainIndex)
