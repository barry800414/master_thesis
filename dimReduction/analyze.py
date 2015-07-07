import sys, pickle
import numpy as np
import matplotlib.pyplot as plt

def printX(X):
    from pprint import pprint
    opt = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    pprint(X)
    np.set_printoptions(**opt)


def printXY(X, y):
    assert X.shape[0] == len(y)
    for i in range(0, len(y)):
        if y[i] == 0:
            continue
        print(i, y[i], end=' ')
        for x in X[i]:
            print('%.3f' % x, end=' ')
        print('')
    print('')
    for i in range(0, len(y)):
        if y[i] == 1: continue
        print(i, y[i], end=' ')
        for x in X[i]:
            print('%.3f' % x, end=' ')
        print('')

def plotX(X, y):
    assert X.shape[0] == len(y)
    posIndex, negIndex = list(), list()
    for i in range(0, len(y)):
        if y[i] == 1:
            posIndex.append(i)
        if y[i] == 0:
            negIndex.append(i)
    
    plot(X[posIndex][:,0], X[posIndex][:,1], 'bo')
    plot(X[negIndex][:,0], X[negIndex][:,1], 'ro')
    plt.show()

def plot(x1, x2, style):
    plt.plot(x1, x2, style)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], 'pickle', file=sys.stderr)
        exit(-1)

    with open(sys.argv[1], 'r+b') as f:
        p = pickle.load(f)
        
    printXY(p['X'], p['y'])
    plotX(p['X'], p['y'])
