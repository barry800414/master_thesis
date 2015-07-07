
import sys, pickle
from RunExperiments import RunExp, ResultPrinter

if __name__ == '__main__':
    if len(sys.argv) != 3 :
        print('Usage:', sys.argv[0], 'pickleFile seedNum', file=sys.stderr)
        exit(-1)
    
    pickleFile = sys.argv[1]
    seedNum = int(sys.argv[2])

    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)
    X, y = p['X'], p['y']
    print(X.shape, file=sys.stderr)
    ResultPrinter.printFirstLine()
    for seed in range(1, seedNum+1):
        RunExp.selfTrainTestNFold(X, y, 'MaxEnt', 'Accuracy', randSeed=seed, test_folds=10, n_folds=10, outfile=sys.stdout)

