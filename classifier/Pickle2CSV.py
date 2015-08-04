
import sys, pickle
from RunExperiments import ResultPrinter


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'pickle csv', file=sys.stderr)
        exit(-1)

with open(sys.argv[1], 'r+b') as f:
    p = pickle.load(f)
    
with open(sys.argv[2], 'w') as f:
    ResultPrinter.printFirstLine(outfile=f)
    for log in p:
        print('SelfTrainTestNFoldWithFC', 'MaxEnt', 'Accuracy', log['model'].toDim(), 1, 1, log['trainScore'], log['valScore'], log['testScore']['Accuracy'], sep=',', file=f)
