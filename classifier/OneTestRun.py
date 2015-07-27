
import sys, pickle
from RunExperiments import RunExp, ResultPrinter
from misc import *

def parseArgument(argv, start):
    pickleFiles = dict()
    fSelectConfig = None
    outLogPickle = None
    for i in range(start, len(argv)):
        if argv[i] == '--fSelect':
            fSelectConfig = { 'params': dict() }
            for j in range(i+1, len(argv)):
                if argv[j] in '-outLogPickle':
                    break
                elif argv[j] == '-method' and len(argv) > i:
                    fSelectConfig['method'] = argv[j+1]
                elif argv[j][0] == '-' and len(argv) > i:
                    try: value = int(argv[j+1])
                    except:
                        try: value = float(argv[j+1])
                        except: value = argv[j+1]
                    fSelectConfig['params'][argv[j][1:]] = value
        elif argv[i] == '-outLogPickle' and len(argv) > i:
            outLogPickle = argv[i+1]
        elif argv[i][0] == '-' and len(sys.argv) > i:
            topic = int(argv[i][1:])
            pickleFiles[topic] = sys.argv[i+1]

    return pickleFiles, outLogPickle, fSelectConfig

def readPickleFiles(pickleFiles):
    finalX = None
    finaly = None
    topicMap = list()
    for topic, filename in pickleFiles.items():
        with open(filename, 'r+b') as f:
            p = pickle.load(f)
        finalX = vMergeX(finalX, p['X'])
        finaly = vMergeX(finaly, p['y'])
        topicMap.extend([topic for i in range(0, len(p['y']))])
    return finalX, finaly, topicMap

if __name__ == '__main__':
    if len(sys.argv) < 3 :
        print('Usage:', sys.argv[0], 'topic seedNum [-topic1 pickle -topic2 pickle ...] [-outLogPickle LogPickle] [--fSelect -method xxx -param1 value1 -param2 value2]', file=sys.stderr)
        exit(-1)
    
    topic = int(sys.argv[1])
    seedNum = int(sys.argv[2])
    pickleFiles, outLogPickle, fSelectConfig = parseArgument(sys.argv, 3)
    print('Test topic:', topic, file=sys.stderr)
    print('pickleFiles:', pickleFiles, file=sys.stderr)
    print('OutLogPickleFile:', outLogPickle, file=sys.stderr)
    print('fSelectConfig:', fSelectConfig, file=sys.stderr)

    X, y, topicMap = readPickleFiles(pickleFiles)
    print(X.shape, file=sys.stderr)
    ResultPrinter.printFirstLine()
    
    logList = list()
    for seed in range(1, seedNum+1):
        logs = RunExp.allTrainOneTestNFold(X, y, topicMap, topic, 'MaxEnt', 'Accuracy', fSelectConfig=fSelectConfig, 
                randSeed=seed, test_folds=10, cv_folds=10, outfile=sys.stdout)
        logList.extend(logs)
    
    if outLogPickle is not None:
        with open(outLogPickle, 'w+b') as f:
            pickle.dump(logList, f)
    
