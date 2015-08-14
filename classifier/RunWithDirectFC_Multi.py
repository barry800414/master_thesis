
import sys, pickle
from FeatureMerge import *
from RunWithFC_Multi import *

from RunExperiments import RunExp, ResultPrinter, DataTool

if __name__ == '__main__':
    if len(sys.argv) < 3 :
        print('Usage:', sys.argv[0], 'seedNum [--inFile pickleFile adjList pickleFile2 adjList2 ... ] [-outLogPickle LogPickle]', file=sys.stderr)
        print('[--fSelect -method xxx -param1 value1 ...] [--preprocess -method xxx -param1 value1 ...]', file=sys.stderr)
        exit(-1)
    
    version = 2
    seedNum = int(sys.argv[1])
    pFileList, aFileList, outLogPickle, fSelectConfig, preprocess = parseArgument(sys.argv, 2)
    print('pFileList:', pFileList, file=sys.stderr)
    print('aFileList:', aFileList, file=sys.stderr)
    print('OutLogPickleFile:', outLogPickle, file=sys.stderr)
    print('fSelectConfig:', fSelectConfig, file=sys.stderr)
    print('preprocess:', preprocess, file=sys.stderr)

    X, y, volc = loadPickleList(pFileList)
    adjSet = readAdjFileList(aFileList)
    assert len(volc) == len(adjSet) 
    checkAdjSet(adjSet)

    ResultPrinter.printFirstLine()

    # feature merge by community detection
    clusters = clusterFeatures(set([i for i in range(0, X.shape[1])]), adjSet) 
    model = genFeatureMergingModel(clusters, list(), volc)
    newX = model.transform(X)
    print('X:', X.shape, ' -> newX:', newX.shape, file=sys.stderr)

    # preprocess if necessary        
    if preprocess is not None:
        newX = DataTool.preprocessX(newX, preprocess['method'], preprocess['params'])

    logList = list()
    for seed in range(1, seedNum+1):
        logs = RunExp.selfTrainTestNFold(newX, y, 'MaxEnt', 'Accuracy', 
                fSelectConfig=fSelectConfig, randSeed=seed, test_folds=10, cv_folds=10, n_jobs=2)
        logs[0]['model'] = model
        logList.extend(logs)
    
    if outLogPickle is not None:
        with open(outLogPickle, 'w+b') as f:
            pickle.dump(logList, f)
   
