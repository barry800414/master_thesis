
import sys, pickle
from scipy.sparse import csr_matrix
from misc import *
from Volc import Volc

def getIDF(df, numDoc):
    idf = dict()
    for w, f in df.items():
        idf[w] = math.log(float(numDoc + 1) / (f + 1))
    return idf

def countColSumByCSRMatrix(X):
    (rowNum, colNum) = X.shape
    (colIndex, rowPtr, data) = X.indices, X.indptr, X.data
    nowPos = 0
    # count for each column
    value = [0 for i in range(0, colNum)] 
    for ri in range(0, rowNum):
        for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
            value[ci] += data[nowPos]
            nowPos += 1
    return value

# X should be the count matrix
def reduce(X, method='tfidf', top=0.3, volc=None):
    topNum = round(X.shape[1] * top) if top <= 1.0 else int(top)
    docNum = X.shape[0]

    if method == 'df':
        value = countDFByCSRMatrix(X)
    elif method == 'tf':
        value = countColSumByCSRMatrix(X)
    if method == 'tfidf':
        DF = countDFByCSRMatrix(X)
        IDF = getIDF(DF, docNum)
        value = countColSumByCSRMatrix(X)
        for i in range(0, len(value)):
            value[i] = value[i] * IDF[i]

    rank = sorted([(i, c) for i, c in enumerate(value)], key=lambda x:x[1], reverse=True)
    mapping = { oldId: newId for newId, (oldId, c) in enumerate(rank[0:topNum]) }
        
    # generate new reduced matrix:
    (rowNum, colNum) = X.shape
    (colIndex, rowPtr, data) = X.indices, X.indptr, X.data
    nowPos = 0
    rows, cols, data = list(), list(), list()
    for ri in range(0, rowNum):
        for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
            if ci in mapping:
                rows.append(ri)
                cols.append(mapping[ci])
                data.append(data[nowPos])
            nowPos += 1

    newX = csr_matrix((data, (rows, cols)), shape=(colNum, len(mapping)))

    if volc is not None:
        newVolc = Volc()
        for w, i in volc.volc.items():
            if i in mapping:
                newVolc[w] = mapping[i]
        assert len(newVolc) == newX.shape[1]
    else:
        newVolc = volc
    return newX, newVolc


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:', sys.argv[0], 'pickle method top', file=sys.stderr)
        exit(-1)

