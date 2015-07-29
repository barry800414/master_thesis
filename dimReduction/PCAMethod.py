
import sys, pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from RunExperiments import RunExp, ResultPrinter

def concatAndRunPCA(pList, n_components=0.95):
    X = None
    for p in pList:
        X = p['X'].todense() if X is None else np.concatenate((X, p['X'].todense()), axis=0)
    pca = PCA(n_components=n_components)
    newX = pca.fit_transform(X)
    print(n_components, newX.shape, file=sys.stderr)
    return pca, newX 

def runPCA(X, n_components=0.95):
    if type(X) == csr_matrix:
        X = X.todense()
    pca = PCA(n_components=n_components)
    newX = pca.fit_transform(X)
    print(n_components, newX.shape, file=sys.stderr)
    return pca, newX 

def splitX(X, pList):
    XList = list()
    nowIndex = 0
    for p in pList:
        XList.append(X[nowIndex: nowIndex + p['X'].shape[0]])
        nowIndex += p['X'].shape[0]
    return XList

def parseNComp(argv, nDocs):
    if argv == 'mle':
        return 'mle'
    elif argv == 'None':
        return nDocs
    else:
        nComp = float(argv)
        nComp = int(nComp) if nComp > 1.0 else nComp
        return nComp

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print('Usage:', sys.argv[0], 'nComponents PicklePrefix1 PicklePrefix2 ....', file=sys.stderr)
        exit(-1)

    nDocs = 0
    pList = list()
    for i in range(2, len(sys.argv)):
        with open(sys.argv[i] + '.pickle', 'r+b') as f:
            p = pickle.load(f)
            nDocs += p['X'].shape[0]
            pList.append(p)
    nComp = parseNComp(sys.argv[1], nDocs)
    print('nComp:', nComp, file=sys.stderr)

    # run pca
    (pca, newX) = concatAndRunPCA(pList, nComp)
    XList = splitX(newX, pList)    
    
    for i, p in enumerate(pList):
        # run pca on unlabeled data
        if p['unX'] is not None:
            newUnX = pca.transform(p['unX'].todense()) if p['unX'].shape[0] > 0 else p['unX']
        else:
            newUnX = None
        pObj = { 'X': XList[i], 'unX': newUnX, 'y': p['y'], 'mainVolc': p['mainVolc'] }
        with open(sys.argv[i+2] + '_PCA%s.pickle' % str(nComp), 'w+b') as f:
            pickle.dump(pObj, f)


