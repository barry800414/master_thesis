
import sys, pickle
from LDA import *
from tfidf import *
from RunExperiments import RunExp, ResultPrinter

if __name__ == '__main__':
    if len(sys.argv) < 4 :
        print('Usage:', sys.argv[0], 'pickleFile reduceMethod usingUnlabeled(0/1) -param1 value1 -param2 value2 ... [-outPickle filename] ', file=sys.stderr)
        exit(-1)
    
    pickleFile = sys.argv[1]
    reduceMethod = sys.argv[2]
    usingUnlabeledData = True if sys.argv[3] == '1' else False
    outPickleFile = None
    param = dict()
    for i in range(4, len(sys.argv)):
        if sys.argv[i] == '-outPickle' and len(sys.argv) > i:
            outPickleFile = sys.argv[i+1]
        if sys.argv[i][0] == '-' and len(sys.argv) > i:
            key = sys.argv[i][1:]
            value = sys.argv[i+1]
            param[key] = value

    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)

    X, y, unX, mainVolc = p['X'], p['y'], p['unX'], p['mainVolc']
    allX = vstack((X, unX)).tocsr() if usingUnlabeledData else X
    if reduceMethod == 'fSelect':
        assert 'method' in param 
        print('Reduction using LDA ... ', end='', file=sys.stderr)
        nT, nIter = float(param['nTopics']), int(param['nIter']) 
        nT = int(nT) if nT >= 1 else round(nT * X.shape[1])
        newAllX = runLDA(allX, nTopics=nT, nIter=nIter)
    elif reduceMethod == 'tfidf':
        assert 'method' in param and 'top' in param
        print('Reduction using %s ... ' % (param['method']), end='', file=sys.stderr)
        method, top = param['method'], float(param['top'])
        top = int(top) if top >= 1 else round(top * X.shape[1])
        newAllX, newVolc = reduce(allX, method, top, mainVolc)
    
    newX = newAllX[0:X.shape[0]] if usingUnlabeledData else newAllX
    newuX = newAllX[X.shape[0]:] if usingUnlabeledData else None

    print(X.shape, ' -> ', newX.shape, file=sys.stderr)
    
    if outPickleFile is not None:
        newP = { 'X': newX, 'y': y, 'unX': newunX, 'mainVolc': newVolc }
        with open(outPickleFile, 'w+b') as f:
            pickle.dump(newP, f)

    # run cross-validation testing
    ResultPrinter.printFirstLine()
    RunExp.selfTrainTestNFold(newX, y, 'MaxEnt', 'Accuracy', randSeed=1, test_folds=10, n_folds=10, outfile=sys.stdout)

