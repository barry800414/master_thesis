import sys
import numpy as np
from RunExperiments import ResultPrinter

def readCSV(filename, dataType=None):
    with open(filename, 'r') as f:
        line = f.readline() # first line: column name
        colNameMap = {c.strip():i for i, c in enumerate(line.strip().split(','))}
        #rows = list()
        cols = [list() for i in range(0, len(dataType))]
        for line in f:
            entry = line.strip().split(',')
            if dataType is not None:
                assert len(dataType) == len(entry)
                #row = list()
                for i, e in enumerate(entry):
                    if dataType[i] == 'int':
                        v = int(e)
                    elif dataType[i] == 'float':
                        v = float(e)
                    elif dataType[i] == 'dict':
                        v = str2Var(e)
                    else:
                        v = e.strip()
                    #row.append(v)
                    cols[i].append(v)
            #else:
                #row = entry
            #rows.append(row)
    return (colNameMap, cols)

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
    (colNameMap, cols) = readCSV(resultCSV, dataType)
    if index is None:
        trainCol, valCol, testCol = 'train', 'val', 'test'
    else:
        trainCol, valCol, testCol = 'train_%d' % (index), 'val_%d' %(index), 'test_%d' %(index)
    #dims = getColumn(rows, colNameMap['dimension'])
    dim = cols[colNameMap['dimension']][0]
    train = np.mean(cols[colNameMap[trainCol]])
    val = np.mean(cols[colNameMap[valCol]])
    test = np.mean(cols[colNameMap[testCol]])
    #trainScores = getColumn(data, colNameMap[trainCol])
    #valScores = getColumn(data, colNameMap[valCol])
    #testScores = getColumn(data, colNameMap[testCol])

    #train = np.mean(trainScores)
    #val = np.mean(valScores)
    #test = np.mean(testScores)
    print(resultCSV, dim, train, val, test, sep=',')
