
import sys, pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from RunExperiments import RunExp, ResultPrinter

def runPCA(pList, nComp, base):
    X = None
    docNum = 0
    sentNum = 0
    for p in pList:
        X = p['sentPX'].todense() if X is None else np.concatenate((X, p['sentPX'].todense()), axis=0)
        docNum += p['docPX'].shape[0]
        sentNum += p['sentPX'].shape[0]
    
    if base == 'doc':
        nComp = int(nComp*docNum)
    elif base == 'sent':
        nComp = int(nComp*sentNum)
    print('base: %s, nComp: %s' % (base, str(nComp)))

    pca = PCA(n_components=nComp)
    newX = pca.fit_transform(X)
    print(newX.shape, file=sys.stderr)
    return pca, newX 

def splitX(X, pList):
    XList = list()
    nowIndex = 0
    for p in pList:
        XList.append(X[nowIndex: nowIndex + p['sentPX'].shape[0]])
        nowIndex += p['sentPX'].shape[0]
    return XList

def parseNComp(argv):
    if argv == 'mle':
        return 'mle'
    elif argv == 'None':
        return None
    else:
        nComp = float(argv)
        nComp = int(nComp) if nComp > 1.0 else nComp
        return nComp

if __name__ == '__main__':
    if len(sys.argv) < 3 :
        print('Usage:', sys.argv[0], 'base(default/doc/sent) nComponents PicklePrefix1 PicklePrefix2 ....', file=sys.stderr)
        exit(-1)

    seedNum = 3
    
    base = sys.argv[1]
    nComp = parseNComp(sys.argv[2])
    pList = list()
    for i in range(3, len(sys.argv)):
        with open(sys.argv[i] + '.pickle', 'r+b') as f:
            pList.append(pickle.load(f))
    
    # run pca on document level feature
    (pca, newSentPX) = runPCA(pList, nComp, base)
    sentPXList = splitX(newSentPX, pList)    
    
    for i, p in enumerate(pList):
        # run pca on sentence feature
        #if p['docPX'] is not None:
        #    newDocPX = pca.transform(p['docPX'].todense()) if p['docPX'].shape[0] > 0 else p['docPX']
        #    print('newDocPX:', newDocPX.shape)
        newDocPX = None
        # run pca on unlabeled data
        #if p['unDocPX'] is not None:
        #    newUnDocPX = pca.transform(p['unDocPX'].todense()) if p['unDocPX'].shape[0] > 0 else p['unDocPX']
        #else:
        newUnDocPX = None
        #if p['unSentPX'] is not None:
        #    newUnSentPX = pca.transform(p['unSentPX'].todense()) if p['unSentPX'].shape[0] > 0 else p['unSentPX']
        #else:
        newUnSentPX = None


        pObj = { 
            'docy': p['docy'], 'senty': p['senty'], 
            'docPX':  newDocPX, 'unDocPX': newUnDocPX, 'sentPX': sentPXList[i], 'unSentPX': newUnSentPX, 
            'sentSX': p['sentSX'], 'unSentSX': p['unSentSX'], 'doc2XList': p['doc2XList'],
            'mainVolc': p['mainVolc']
        }
        
        with open(sys.argv[i+3] + '_PCA%s.pickle' % str(nComp), 'w+b') as f:
            pickle.dump(pObj, f)


