
import sys, pickle
from RunExp import RunExp, ResultPrinter, DataTool
from FeatureMerge import *

def parseArgument(argv, start):
    fSelectConfig = None
    outLogPickle = None
    preprocess = None
    nClusterDict = None
    for i in range(start, len(argv)):
        if argv[i] == '--fSelect':
            fSelectConfig = { 'params': dict() }
            for j in range(i+1, len(argv)):
                if argv[j] in ['-outLogPickle', '--preprocess', '--nCluster']:
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
                if argv[j] in ['-outLogPickle', '--fSelect', '--nCluster']:
                    break
                elif argv[j] == '-method' and len(argv) > i:
                    preprocess['method'] = argv[j+1]
                elif argv[j][0] == '-' and len(argv) > i:
                    try: value = int(argv[j+1])
                    except:
                        try: value = float(argv[j+1])
                        except: value = argv[j+1]
                    preprocess['params'][argv[j][1:]] = value

        elif argv[i] == '--nCluster':
            nClusterDict = dict()
            for j in range(i+1, len(argv), 2):
                if argv[j] in ['-outLogPickle', '--preprocess', '--fSelect']:
                    break
                else:
                    nClusterDict[argv[j]] = float(argv[j+1])
        elif argv[i] == '-outLogPickle' and len(argv) > i:
            outLogPickle = argv[i+1]

    return outLogPickle, fSelectConfig, preprocess, nClusterDict

if __name__ == '__main__':
    if len(sys.argv) < 6 :
        print('Usage:', sys.argv[0], 'pickleFile wordVectorFile FeatureGroupVersion nClusters seedNum [-outLogPickle LogPickle]', file=sys.stderr)
        print('[--nCluster group1 nCluster1 group2 nCluster2 ...] ', file=sys.stderr)
        print('[--fSelect -method xxx -param1 value1 ...] [--preprocess -method xxx -param1 value1 ...] ', file=sys.stderr)
        exit(-1)
    
    pickleFile = sys.argv[1]
    wordVectorFile = sys.argv[2]
    featureGroupVersion = int(sys.argv[3])
    nClusters = float(sys.argv[4])
    seedNum = int(sys.argv[5])
    outLogPickle, fSelectConfig, preprocess, nClusterDict = parseArgument(sys.argv, 5)
    if nClusterDict is None:
        nClusterDict = nClusters

    print('OutLogPickleFile:', outLogPickle, file=sys.stderr)
    print('fSelectConfig:', fSelectConfig, file=sys.stderr)
    print('preprocess:', preprocess, file=sys.stderr)
    print('nClusterDict:', nClusterDict, file=sys.stderr)

    # read word vectors
    volc, vectors = readWordVector(wordVectorFile)
    wordVector = toDictType(volc, vectors)
    
    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)
    X, y, volc = p['X'], p['y'], p['mainVolc']
    ResultPrinter.printFirstLine()

    # convert it to feature vectors
    groupVectors, groupVolc, groupMapping = getFeatureVectorsByGroup(volc, wordVector, featureGroupVersion)
    
    # feature merging using Kmeans
    model = featureClustering_KMeans_byGroup(groupVectors, groupMapping, nClusterDict, max_iter=100)
    newX = model.transform(X)
    print('X:', X.shape, ' -> newX:', newX.shape, file=sys.stderr)
    # preprocess if necessary        
    if preprocess is not None:
        newX = DataTool.preprocessX(newX, preprocess['method'], preprocess['params'])
    logList = list()
    for seed in range(1, seedNum+1):
        logs = RunExp.selfTrainTestNFold(newX, y, 'MaxEnt', 'Accuracy', 
                fSelectConfig=fSelectConfig, randSeed=seed, test_folds=10, cv_folds=10, n_jobs=2)
        logList.extend(logs)
    
    if outLogPickle is not None:
        with open(outLogPickle, 'w+b') as f:
            pickle.dump(logList, f)
    
