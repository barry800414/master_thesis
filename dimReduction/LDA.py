#!/usr/bin/env python3
import sys, json, pickle
from collections import defaultdict

import numpy as np
import lda
from scipy.sparse import csr_matrix, vstack

from misc import *
from Volc import *

# now only support csr_matrix
def concatAndRunLDA(pList, nTopic, nIter):
    X = None
    volc = None
    for p in pList:
        X = p['X'] if X is None else vstack((X, p['X'])).tocsr()
        volc = p['mainVolc'] if volc is None else volc

    (model, newVolc) = runLDA(X, volc, nTopic, nIter)
    newX = model.doc_topic_
    return model, newX, newVolc 

def splitX(X, pList):
    XList = list()
    nowIndex = 0
    for p in pList:
        XList.append(X[nowIndex: nowIndex + p['X'].shape[0]])
        nowIndex += p['X'].shape[0]
    return XList

# vocab is a list (index -> word mapping)
def runLDA(W, volc=None, nTopics=10, nIter=10, nTopicWords=100, randomState=1, outfile=sys.stdout):
    if nTopicWords == -1:
        nTopicWords = len(vocab) # all words
    model = lda.LDA(n_topics=nTopics, n_iter=nIter, random_state=randomState)
    model.fit(W)
    if volc is not None:
        vocab = geti2W(volc)
        topicWord = model.topic_word_
        newVolc = Volc() # use top N words as volcabulary
        for t, wordDist in enumerate(topicWord): # for each topic select top N words
            ipList = [(i, p) for i, p in enumerate(wordDist)] # list of index, prob tuples
            ipList.sort(key = lambda x:x[1], reverse=True)
            topicWords = tuple([volc.getWord(i) for i, p in ipList[0:nTopicWords]])
            newVolc.addWord(topicWords)
        return model, newVolc
    else:
        return model

def geti2W(volc):
    if volc is None:
        return None
    i2w = [volc.getWord(i, usingJson=True) for i in range(0, len(volc))]
    return i2w

# print Topic-Word Matrix [topicNum x wordNum] (phi in literature)
def printTWMatrix(model, i2w, encoding='utf-8', outfile=sys.stdout):
    for w in i2w:
        outfile.write((w + ',').encode(encoding))
    np.savetxt(outfile, model.topic_word_, delimiter=',')

def parseNTopic(argv, nDocs):
    if argv == 'None':
        return nDocs
    else:
        nT = float(argv)
        nT = int(nT) if nT > 1.0 else int(nT * nDocs)
        return nT

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print('Usage:', sys.argv[0], 'nTopic nIter PicklePrefix1 PicklePrefix2 ....', file=sys.stderr)
        exit(-1)

    nIter = int(sys.argv[2])
    nDocs = 0
    pList = list()
    for i in range(3, len(sys.argv)):
        with open(sys.argv[i] + '.pickle', 'r+b') as f:
            p = pickle.load(f)
            pList.append(p)
            nDocs += p['X'].shape[0]
    nTopic = parseNTopic(sys.argv[1], nDocs)
    print('nTopic:', nTopic, 'nIter:', nIter, file=sys.stderr)

    # run lda
    (model, newX, newVolc) = concatAndRunLDA(pList, nTopic, nIter)
    XList = splitX(newX, pList)    
    
    for i, p in enumerate(pList):
        # run lda on unlabeled data
        #if p['unX'] is not None:
        #    newUnX = model.transform(p['unX']) if p['unX'].shape[0] > 0 else p['unX']
        #else:
        #   newUnX = None
        pObj = { 'X': XList[i], 'unX': None, 'y': p['y'], 'mainVolc': newVolc }
        with open(sys.argv[i+3] + '_LDA%s.pickle' % str(nTopic), 'w+b') as f:
            pickle.dump(pObj, f)

