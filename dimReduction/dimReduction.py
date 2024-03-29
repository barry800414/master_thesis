
import sys, pickle
from scipy.sparse import vstack
from LDA import *
from tfidf import *
from PCAMethod import *
from RunExperiments import RunExp, ResultPrinter

if __name__ == '__main__':
    if len(sys.argv) < 4 :
        print('Usage:', sys.argv[0], 'pickleFile reduceMethod usingUnlabeled(0/1) -param1 value1 -param2 value2 ... [-outPickle filename] [-outLogPickle filename] [-run seedNum]', file=sys.stderr)
        exit(-1)
    
    pickleFile = sys.argv[1]
    reduceMethod = sys.argv[2]
    usingUnlabeledData = True if sys.argv[3] == '1' else False
    outPickleFile = None
    outLogPickleFile = None
    toRun = False
    param = dict()
    for i in range(4, len(sys.argv)):
        if sys.argv[i] == '-outPickle' and len(sys.argv) > i:
            outPickleFile = sys.argv[i+1]
        elif sys.argv[i] == '-outLogPickle' and len(sys.argv) > i:
            outLogPickleFile = sys.argv[i+1]
        elif sys.argv[i] == '-run' and len(sys.argv) > i:
            toRun = True
            seedNum = int(sys.argv[i+1])
        elif sys.argv[i][0] == '-' and len(sys.argv) > i:
            key = sys.argv[i][1:]
            value = sys.argv[i+1]
            param[key] = value

    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)

    # preparign data
    X, y, unX, mainVolc = p['X'], p['y'], p['unX'], p['mainVolc']
    allX = vstack((X, unX)).tocsr() if usingUnlabeledData else X
    model = None

    # reducing dimension 
    if reduceMethod == 'LDA':
        assert 'nTopics' in param and 'nIter' in param
        nT, nIter = float(param['nTopics']), int(param['nIter']) 
        nT = int(nT) if nT > 1.0 else round(nT * X.shape[1])
        print('Reduction using LDA ... #topics:%d ' % (nT), end='', file=sys.stderr)
        (model, newVolc) = runLDA(allX, volc=mainVolc, nTopics=nT, nIter=nIter) #the model
        newAllX = model.doc_topic_

    elif reduceMethod == 'PCA':
        assert 'nComp' in param
        nComp = parseNComp(param['nComp'], X.shape[0])
        print('Reduction using PCA ... nComp:%s' % (nComp), file=sys.stderr)
        (model, newAllX) = runPCA(allX, nComp)
        newVolc = None
        unX = unX.todense()

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
        newX, newunX = newAllX[0:X.shape[0]], newAllX[X.shape[0]:]  #could cause problem
    else: # otherwise transform it if there is transformer
        newX = newAllX
        print('Reduction on unlabeled data ...', file=sys.stderr)
        newunX = model.transform(unX) if model is not None else None
    
    # print shape information
    print(X.shape, ' -> ', newX.shape, file=sys.stderr)
    if newunX is not None:
        print(unX.shape, ' -> ', newunX.shape, file=sys.stderr)
    else:
        print(unX.shape, ' -> None', file=sys.stderr)

    # output data 
    if outPickleFile is not None:
        newP = { 'X': newX, 'y': y, 'unX': newunX, 'mainVolc': newVolc }
        with open(outPickleFile, 'w+b') as f:
            pickle.dump(newP, f)
    
    # run cross-validation testing
    if toRun:
        logList = list()
        ResultPrinter.printFirstLine()
        for seed in range(1, seedNum+1):
            logs = RunExp.selfTrainTestNFold(newX, y, 'MaxEnt', 'Accuracy', randSeed=seed, test_folds=10, cv_folds=10, outfile=sys.stdout)
        logList.extend(logs)
        if outLogPickleFile is not None:
            with open(outLogPickleFile, 'w+b') as f:
                pickle.dump(logList, f)
            

