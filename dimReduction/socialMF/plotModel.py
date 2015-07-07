
import matplotlib.pyplot as plt

# X: document latent matrix (Dxk)
# y: document label (Dx1)
def plotModel(X, y, trainIndex, testIndex):
    trainSet = set(trainIndex)
    testSet = set(testIndex)
    index = { 'train': { 'pos': list(), 'neg': list() }, 
              'test': { 'pos': list(), 'neg': list()} }
    for i in range(0, len(y)):
        if i in trainSet: name = 'train'
        else: name = 'test'
        if y[i] == 1: label = 'pos'
        else: label = 'neg'
        index[name][label].append(i)

    # pos in train
    i = index['train']['pos']
    plt.plot(X[i][:,0], X[i][:,1], 'ro')
    # neg in train
    i = index['train']['neg']
    plt.plot(X[i][:,0], X[i][:,1], 'bo')
    # pos in test
    i = index['test']['pos']
    plt.plot(X[i][:,0], X[i][:,1], 'r.')
    # neg in test
    i = index['test']['neg']
    plt.plot(X[i][:,0], X[i][:,1], 'b.')
    plt.show()


