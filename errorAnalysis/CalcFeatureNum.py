
import sys, pickle
import numpy as np
from scipy.sparse import vstack


def calcFeatureNum(X):
    (rowNum, colNum) = X.shape
    colIndex, rowPtr, data = X.indices, X.indptr, X.data
    fNum = list()
    for ri in range(0, rowNum):
        fNum.append(rowPtr[ri+1] - rowPtr[ri])
    return np.array(fNum)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:', sys.argv[0], 'pickleType(data/log) pickle', file=sys.stderr) 
        exit(-1)

    pickleType = sys.argv[1]
    with open(sys.argv[2], 'r+b') as f:
        p = pickle.load(f)


    if pickleType == 'log':
        X = vstack((p[0]['XTrain'], p[0]['XTest'])).tocsr()
    elif pickleType == 'data':
        X = p['X']

    fNum = calcFeatureNum(X)
    # average feature num/ average feature percentage / nDocNoF / nDocNoFPercent
    nDocNoF = X.shape[0] - np.count_nonzero(fNum)
    nDocNoFPercent = 100*(nDocNoF) / X.shape[0]
    print(sys.argv[2], np.mean(fNum), np.mean(fNum / X.shape[1]), nDocNoF, nDocNoFPercent, sep=',')

