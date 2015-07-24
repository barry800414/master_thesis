
import sys, pickle
from RunExperiments import RunExp, ResultPrinter

if __name__ == '__main__':
    if len(sys.argv) < 3 :
        print('Usage:', sys.argv[0], 'pickleFile seedNum [LogPickle]', file=sys.stderr)
        exit(-1)
    
    pickleFile = sys.argv[1]
    seedNum = int(sys.argv[2])
    logPickleFile = sys.argv[3] if len(sys.argv) == 4 else None

    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)
    X, y = p['X'], p['y']
    print(X.shape, file=sys.stderr)
    ResultPrinter.printFirstLine()
    
    logList = list()
    for seed in range(1, seedNum+1):
        logs = RunExp.selfTrainTestNFold(X, y, 'MaxEnt', 'Accuracy', randSeed=seed, test_folds=10, cv_folds=10, outfile=sys.stdout)
        logList.extend(logs)
    
    if logPickleFile is not None:
        with open(logPickleFile, 'w+b') as f:
            pickle.dump(logList, f)

