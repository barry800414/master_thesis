
import sys, pickle
from scipy.sparse import vstack
from LDA import *
from tfidf import *
from RunExperiments import RunExp, ResultPrinter

'''
This version is for running dimension reduction techniques for several topics together (with shared volcabulary)
Date: 2015/07/10
'''

def prepareData(pList):
    mergedX, mergedy = None, None
    for p in pList:
        X = p['X']
        mergedX = X if mergedX is None else vstack((mergedX, X)).tocsr()
    mergedUnX = None
    for p in pList:
        unX = p['unX']
        mergedUnX = X if mergedUnX is None else vstack((mergedUnX, unX)).tocsr()
    return mergedX, mergedUnX

def splitData(allX, pList, dataType):
    nowIndex = 0
    if dataType in ['label', 'both']:
        XList = list()
        nLabelDoc = [p['X'].shape[0] for p in pList]    
        for nDoc in nLabelDoc:
            XList.append(allX[nowIndex:nowIndex + nDoc])
            nowIndex += nDoc
        
    if dataType in ['unlabel', 'both']:
        nUnlabelDoc = [p['unX'].shape[0] for p in pList]
        unXList = list()
        for nDoc in nUnlabelDoc:
            unXList.append(allX[nowIndex:nowIndex + nDoc])
    
    if dataType == 'label':
        return XList
    elif dataType == 'unlabel':
        return unXList
    else:
        return XList, unXList

def splitY(ally, pList):
    nowIndex = 0
    yList = list()
    nLabelDoc = [p['y'].shape[0] for p in pList]    
    for nDoc in nLabelDoc:
        yList.append(ally[nowIndex:nowIndex + nDoc])
        nowIndex += nDoc
    return yList

if __name__ == '__main__':
    if len(sys.argv) < 5 :
        print('Usage:', sys.argv[0], 'reduceMethod usingUnlabeled(0/1) outPickleFilePrefix -inPickle name pickleFile1 ... -param1 value1 -param2 value2 ...', file=sys.stderr)
        exit(-1)
    
    reduceMethod = sys.argv[1]
    usingUnlabeledData = True if sys.argv[2] == '1' else False
    outPicklePrefix = sys.argv[3]
    pList, pNameList = list(), list()
    param = dict()
    for i in range(4, len(sys.argv)):
        if sys.argv[i] == '-inPickle' and len(sys.argv) > i+1:
            pNameList.append(sys.argv[i+1])
            with open(sys.argv[i+2], 'r+b') as f:
                pList.append(pickle.load(f))
        elif sys.argv[i][0] == '-' and len(sys.argv) > i:
            key = sys.argv[i][1:]
            value = sys.argv[i+1]
            param[key] = value

    X, unX = prepareData(pList)
    allX = vstack((X, unX)).tocsr() if usingUnlabeledData else X
    mainVolc = pList[0]['mainVolc']
    model = None
    print('Original X:', X.shape, ' unX:', unX.shape, file=sys.stderr)

    # reducing dimension 
    if reduceMethod == 'LDA':
        assert 'nTopics' in param and 'nIter' in param
        print('Reduction using LDA ... ', end='', file=sys.stderr)
        nT, nIter = float(param['nTopics']), int(param['nIter']) 
        nT = int(nT) if nT > 1.0 else round(nT * X.shape[1])
        model, newVolc = runLDA(allX, mainVolc, nTopics=nT, nIter=nIter) #the model
        newAllX = model.doc_topic_

    elif reduceMethod == 'tfidf':
        assert 'method' in param and ('top' in param or 'minCnt' in param)
        print('Reduction using %s ... ' % (param['method']), end='', file=sys.stderr)
        method = param['method']
        if 'top' in param:
            top = int(param['top']) if float(param['top']) > 1.0 else round(float(param['top']) * X.shape[1])
            newAllX, newVolc, model = reduce(allX, method, top, mainVolc, notRemoveRow=True)
        if 'minCnt' in param:
            minCnt = int(param['minCnt'])
            newAllX, newVolc, model = reduceByDF(allX, minCnt, mainVolc, notRemoveRow=True)
    
    # if using unlabeled data, split it
    if usingUnlabeledData:
        newXList, newUnXList = splitData(newAllX, pList, 'both')
    else: # otherwise transform it if there is transformer
        newXList = splitData(newAllX, pList, 'label')
        print('Reduction on unlabeled data ...', file=sys.stderr)
        if reduceMethod == 'LDA':
            newUnX = model.transform(unX) if model is not None else None
        elif reduceMethod == 'tfidf':
            newUnX = model.transform(unX, notRemoveRow=True) if model is not None else None
        newUnXList = splitData(newUnX, pList, 'unlabel')

    # output data 
    assert len(newXList) == len(newUnXList) and len(newXList) == len(pList)
    for i, p in enumerate(pList):
        assert p['y'].shape[0] == newXList[i].shape[0]
        newP = { 'X': newXList[i], 'y': p['y'], 'unX': newUnXList[0], 'mainVolc': newVolc }
        with open('%s_%s.pickle' % (outPicklePrefix, pNameList[i]), 'w+b') as f:
            pickle.dump(newP, f)
        print(pNameList[i], ': X:', newXList[i].shape, 'unX:', newUnXList[i].shape, file=sys.stderr)
    
