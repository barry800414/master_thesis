
import sys,pickle

from scipy.sparse import hstack, csr_matrix
import numpy as np
from Volc import Volc

def mergeX(x1, x2):
    newX = None
    if type(x1) == csr_matrix:
        if type(x2) == csr_matrix:
            newX = hstack((x1, x2)).tocsr()
        elif type(x2) == np.ndarray:
            n1 = x1.shape[0] * x1.shape[1]
            n2 = x2.shape[0] * x2.shape[1]
            if n1 > n2:
                newX = hstack((x1, csr_matrix(x2))).tocsr()
            else:
                newX = np.concatenate((x1.todense(), x2), axis=1)
        else:
            print('Type: ', type(x2), 'not support', file=sys.stderr)
    elif type(x1) == np.ndarray:
        if type(x2) == np.ndarray:
            newX = np.concatenate((x1, x2), axis=1)
        elif type(x2) == csr_matrix:
            n1 = x1.shape[0] * x1.shape[1]
            n2 = x2.shape[0] * x2.shape[1]
            if n1 > n2:
                newX = np.concatenate((x1, x2.todense()), axis=1)
            else:
                newX = hstack((csr_matrix(x1), x2)).tocsr()
        else:
            print('Type: ', type(x2), 'not support', file=sys.stderr)
    else:
        print('Type: ', type(x1), 'not support', file=sys.stderr)
    return newX

def mergeVolc(v1, v2):
    newVolc = v1.copy(lock=False)
    offset = len(newVolc)
    for w in v2.volc.keys():
        newVolc[w] = v2[w] + offset
    return newVolc

def mergePickle(pickleList, mergeUnX=True):
    X = None
    unX = None
    y = None
    mainVolc = None
    for p in pickleList:
        print('X:', p['X'].shape, 'y:', p['y'].shape, 'mainVolc:', len(p['mainVolc']), file=sys.stderr)
        if mergeUnX and unX is not None:
            print('unX:', unX.shape, file=sys.stderr)

        if X is None: X = p['X']
        else: X = mergeX(X, p['X'])
        
        if mergeUnX:
            if unX is None: unX = p['unX']
            else: unX = mergeX(unX, p['unX'])
        
        if y is None: y = p['y']
        else: assert np.array_equal(y, p['y'])
        
        if mainVolc is None: mainVolc = p['mainVolc']
        else: mainVolc = mergeVolc(mainVolc, p['mainVolc'])
    
    print('Final X:', X.shape, 'Final y:', y.shape, 'Final mainVolc:', len(mainVolc), file=sys.stderr)
    if unX is not None:
        print('Final unX:', unX.shape, file=sys.stderr)

    newP = { 'X': X, 'unX': unX, 'y': y, 'mainVolc': mainVolc }

    return newP


if __name__ == '__main__':
    if len(sys.argv) < 5 :
        print('Usage:', sys.argv[0], 'outPickle mergeUnX(0/1) pickle1 pickle2 ...', file=sys.stderr)
        exit(-1)

    outPickleFile = sys.argv[1]
    assert sys.argv[2] in ['0', '1']
    mergeUnX = True if sys.argv[2] == '1' else False
    pickleList = list()
    for i in range(3, len(sys.argv)):
        with open(sys.argv[i], 'r+b') as f:
            pickleList.append(pickle.load(f))

    newP = mergePickle(pickleList, mergeUnX)

    with open(outPickleFile, 'w+b') as f:
        pickle.dump(newP, f)

