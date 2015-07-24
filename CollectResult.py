import sys
import numpy as np
from RunExperiments import ResultPrinter

def readCSV(filename, dataType=None):
    with open(filename, 'r') as f:
        line = f.readline() # first line: column name
        colNameMap = {c.strip():i for i, c in enumerate(line.strip().split(','))}

        data = list()
        for line in f:
            entry = line.strip().split(',')
            if dataType is not None:
                assert len(dataType) == len(entry)
                row = list()
                for i, e in enumerate(entry):
                    if dataType[i] == 'int':
                        row.append(int(e))
                    elif dataType[i] == 'float':
                        row.append(float(e))
                    elif dataType[i] == 'dict':
                        row.append(str2Var(e))
                    else:
                        row.append(e.strip())
            else:
                row = entry
            data.append(row)
    return (colNameMap, data)

def getColumn(data, i):
    return [d[i] for d in data]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], 'ResultCSV [index]', file=sys.stderr)
        exit(-1)
    
    resultCSV = sys.argv[1]
    index = None
    if len(sys.argv) == 3:
        index = int(sys.argv[2])

    dataType = ResultPrinter.getDataType()
    (colNameMap, data) = readCSV(resultCSV, dataType)
    
    if index is None:
        trainCol, valCol, testCol = 'train', 'val', 'test'
    else:
        trainCol, valCol, testCol = 'train_%d' % (index), 'val_%d' %(index), 'test_%d' %(index)
    dims = getColumn(data, colNameMap['dimension'])
    trainScores = getColumn(data, colNameMap[trainCol])
    valScores = getColumn(data, colNameMap[valCol])
    testScores = getColumn(data, colNameMap[testCol])

    train = np.mean(trainScores)
    val = np.mean(valScores)
    test = np.mean(testScores)
    print(resultCSV, dims[0], train, val, test, sep=',')
