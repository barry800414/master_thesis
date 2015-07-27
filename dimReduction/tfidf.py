
import sys, pickle
from scipy.sparse import csr_matrix
from misc import *
from Volc import Volc

def getIDF(df, numDoc):
    idf = [math.log(float(numDoc + 1) / (df[i] + 1)) for i in range(0, len(df))]
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
def reduce(X, method='tfidf', top=0.3, volc=None, returnRemovedRows=False, notRemoveRow=False):
    topNum = round(X.shape[1] * top) if top < 1.0 else int(top)
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
    transformer = Transformer(mapping)
    newVolc = transformer.transformVolc(volc)
    if returnRemovedRows:
        newX, removedRows = transformer.transform(X, True, notRemoveRow)
    else:
        newX = transformer.transform(X, False, notRemoveRow)
    assert len(newVolc) == newX.shape[1]
    if returnRemovedRows:
        return newX, newVolc, transformer, removedRows
    else:
        return newX, newVolc, transformer

def reduceByDF(X, minCnt=5, volc=None, returnRemovedRows=False, notRemoveRow=False):
    DF = countDFByCSRMatrix(X)
    mapping = dict()
    for i in range(0, len(DF)):
        if DF[i] >= minCnt:
            mapping[i] = len(mapping)
    transformer = Transformer(mapping)
    if returnRemovedRows:
        newX, removedRows = transformer.transform(X, True, notRemoveRow)
    else:
        newX = transformer.transform(X, False, notRemoveRow)
    newVolc = transformer.transformVolc(volc)
    assert len(newVolc) == newX.shape[1]
    if returnRemovedRows:
        return newX, newVolc, transformer, removedRows
    else:
        return newX, newVolc, transformer
    
class Transformer():
    def __init__(self, mapping):
        self.mapping = mapping

    def transform(self, X, returnRemovedRows=False, notRemoveRow=True):    
        if notRemoveRow:
            rowMapping = {i:i for i in range(0, X.shape[0])}
        else:
            # traverse for first time 
            (rowNum, colNum) = X.shape
            (colIndex, rowPtr, data) = X.indices, X.indptr, X.data
            rowHasValue = [False for i in range(0, rowNum)]
            for ri in range(0, rowNum):
                for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
                    if ci in self.mapping:
                        rowHasValue[ri] = True
            rowMapping = dict()
            removedRows = list()
            for i in range(0, rowNum):
                if not rowHasValue[i]:
                    removedRows.append(i)
                    print('Warning: document %d has no value after removal!' % (i), file=sys.stderr)
                else:
                    rowMapping[i] = len(rowMapping)

        # generate new reduced matrix:
        (rowNum, colNum) = X.shape
        (colIndex, rowPtr, data) = X.indices, X.indptr, X.data
        nowPos = 0
        rows, cols, newData = list(), list(), list()
        colHasValue = [False for i in range(0, rowNum)]
        for ri in range(0, rowNum):
            if rowMapping is not None and ri not in rowMapping:
                continue
            for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
                if ci in self.mapping:
                    rows.append(rowMapping[ri])
                    cols.append(self.mapping[ci])
                    newData.append(data[nowPos])
                nowPos += 1
        
        newX = csr_matrix((newData, (rows, cols)), shape=(len(rowMapping), len(self.mapping)), dtype=X.dtype)
        if returnRemovedRows:
            return newX, removedRows
        else:
            return newX

    def transformVolc(self, volc):
        if volc is not None:
            newVolc = Volc()
            for w, i in volc.volc.items():
                if i in self.mapping:
                    newVolc[w] = self.mapping[i]
        else:
            newVolc = volc
        return newVolc

