
import sys, pickle

# This module is to print feature clusters to see the physical meaning

def getFeatureClusters(projMatrix, volc):
    assert projMatrix.shape[0] == len(volc)

    X = projMatrix
    (rowNum, colNum) = X.shape
    (colIndex, rowPtr, data) = (X.indices, X.indptr, X.data)
    
    clusters = [list() for i in range(0, X.shape[1])]
    nowPos = 0
    for ri in range(0, rowNum):
        for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
            if data[nowPos] > 0:
                clusters[ci].append(volc.getWord(ri))

    return clusters


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'PickleFile FeaturePickle', file=sys.stderr)
        exit(-1)
        
    with open(sys.argv[1], 'r+b') as f:
        p = pickle.load(f)
    projMatrix = p[0]['model'].proj
    with open(sys.argv[2], 'r+b') as f:
        p2 = pickle.load(f)       
    volc = p2['mainVolc']
    
    clusters = getFeatureClusters(projMatrix, volc)
    
    for c in clusters:
        print(c)
