
import sys, pickle
from scipy.sparse import hstack
from RunExp_FM import RunExp, ResultPrinter
from FeatureMerge import *

'''
Filename 'Multi' means this module can accept different threshold for different group of features
'''

def parseArgument(argv, start):
    fSelectConfig = None
    outLogPickle = None
    preprocess = None
    for i in range(start, len(argv)):
        if argv[i] == '--fSelect':
            fSelectConfig = { 'params': dict() }
            for j in range(i+1, len(argv)):
                if argv[j] in ['-outLogPickle', '--preprocess', '--inFile']:
                    break
                elif argv[j] == '-method' and len(argv) > i:
                    fSelectConfig['method'] = argv[j+1]
                elif argv[j][0] == '-' and len(argv) > i:
                    try: value = int(argv[j+1])
                    except:
                        try: value = float(argv[j+1])
                        except: value = argv[j+1]
                    fSelectConfig['params'][argv[j][1:]] = value
        elif argv[i] == '--preprocess':
            preprocess = { 'params': dict() }
            for j in range(i+1, len(argv)):
                if argv[j] in ['-outLogPickle', '--fSelect', '--inFile']:
                    break
                elif argv[j] == '-method' and len(argv) > i:
                    preprocess['method'] = argv[j+1]
                elif argv[j][0] == '-' and len(argv) > i:
                    try: value = int(argv[j+1])
                    except:
                        try: value = float(argv[j+1])
                        except: value = argv[j+1]
                    preprocess['params'][argv[j][1:]] = value
        elif argv[i] == '--inFile':
            pFileList = list()
            aFileList = list()
            for j in range(i+1, len(argv), 2):
                if argv[j] in ['-outLogPickle', '--preprocess', '--fSelect']:
                    break
                else:
                    pFileList.append(argv[j])
                    aFileList.append(argv[j+1])

        elif argv[i] == '-outLogPickle' and len(argv) > i:
            outLogPickle = argv[i+1]

    return pFileList, aFileList, outLogPickle, fSelectConfig, preprocess


def readAdjFileList(aFileList):
    finalAdjSet = list()
    offset = 0
    for filename in aFileList:
        adjSet = readAdjList(filename, offset)
        offset += len(adjSet)
        finalAdjSet.extend(adjSet)
    return finalAdjSet

def checkAdjSet(adjSet):
    for i, s in enumerate(adjSet):
        for e in s:
            assert e > i

def loadPickleList(pFileList):
    X = None
    y = None
    volc = None
    for filename in pFileList:
        with open(filename, 'r+b') as f:
            p = pickle.load(f)
        if y is not None:
            assert np.array_equal(y, p['y'])
        else:
            y = p['y']
        
        X = p['X'] if X is None else hstack((X, p['X'])).tocsr()
        volc = p['mainVolc'] if volc is None else Volc.mergeVolc(volc, p['mainVolc'])

    assert X.shape[0] == y.shape[0] and len(volc) == X.shape[1]
    return X, y, volc

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

    logList = list()
    for seed in range(1, seedNum+1):
        logs = RunExp.selfTrainTestNFoldWithFC(version, X, y, volc, adjSet, 'MaxEnt', 'Accuracy', 
                fSelectConfig=fSelectConfig, preprocess=preprocess, randSeed=seed, 
                test_folds=10, cv_folds=10, n_jobs=2)
        logList.extend(logs)
    
    if outLogPickle is not None:
        with open(outLogPickle, 'w+b') as f:
            pickle.dump(logList, f)
    
