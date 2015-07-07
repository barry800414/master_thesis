
import sys, pickle
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine, correlation

def printCSRMatrix(X, outfile=sys.stdout):
    (rowNum, colNum) = X.shape
    (colIndex, rowPtr, data) = X.indices, X.indptr, X.data
    nowPos = 0
    for ri in range(0, rowNum):
        for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
            print(ri, ci, data[nowPos], file=outfile)
            nowPos += 1

def printEdgeList(edgeList, outfile=sys.stdout):
    for (i, j, sim) in edgeList:
        print(i, j, sim, file=outfile)

def calcDocSim(X, y, simType):
    assert X.shape[0] == len(y)

    edgeList = list()
    if simType == 'labelOnly':
        for i in range(0, len(y)):
            for j in range(0, len(y)):
                if i == j: continue
                if y[i] == y[j]:
                    edgeList.append((i, j, 1.0))
                else:
                    edgeList.append((i, j, -1.0))

    # actually considering label and cosine similarity
    elif simType == 'cosine':         
        for i in range(0, len(y)):
            for j in range(0, len(y)):
                if i == j: continue
                if y[i] == y[j]:
                    # (1 + cos(x1, x2) / 2
                    edgeList.append((i, j, ((1.0 + calcCosSim(X.getrow(i), X.getrow(j))) / 2.0)))

    # actually considering label and pearson correlation
    elif simType == 'pearson': 
        for i in range(0, len(y)):
            for j in range(0, len(y)):
                if i == j: continue
                if y[i] == y[j]:
                    # (1 + pearson(x1, x2) / 2
                    edgeList.append((i, j, ((1.0 + calcPearsonSim(X.getrow(i), X.getrow(j)) / 2.0))))

    return edgeList


def calcCosSim(x1, x2):
    if type(x1) == csr_matrix:
        x1 = x1.todense()
    if type(x2) == csr_matrix:
        x2 = x2.todense()
    return 1.0 - cosine(x1, x2)

def calcPearsonSim(x1, x2):
    if type(x1) == csr_matrix:
        x1 = x1.todense()
    if type(x2) == csr_matrix:
        x2 = x2.todense()
    return 1.0 - correlation(x1, x2)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage:', sys.argv[0], 'PickleFile [-outX outXFile] [-outY outYFile][-outNetwork outNetworkFile Similarity(cosine/pearson/labelOnly)]', file=sys.stderr)
        exit(-1);

    pickleFile = sys.argv[1]
    simType = None
    outXFile = None
    outYFile = None
    outNetworkFile = None
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '-outX' and len(sys.argv) > i:
            outXFile = sys.argv[i+1]
        if sys.argv[i] == '-outY' and len(sys.argv) > i:
            outYFile = sys.argv[i+1]
        if sys.argv[i] == '-outNetwork' and len(sys.argv) > (i+1):
            outNetworkFile = sys.argv[i+1]
            simType = sys.argv[i+2]
    

    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)

    X, y = p['X'], p['y']
    
    # generating matrix file for social regularization MF
    if outXFile is not None:
        print('Output X file ...', file=sys.stderr)
        with open(outXFile, 'w') as f:
            printCSRMatrix(X, f)

    # generating social network file
    if outNetworkFile is not None and simType is not None:
        print('Output network file ...', file=sys.stderr)
        edgeList = calcDocSim(X, y, simType)
        with open(outNetworkFile, 'w') as f:
            printEdgeList(edgeList, f)

    if outYFile is not None:
        with open(outYFile, 'w') as f:
            for i in y:
                print(i, file=f)

