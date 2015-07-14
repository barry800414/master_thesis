#!/usr/bin/env python3
import sys
from collections import defaultdict
import json

import numpy as np
import lda
from scipy.sparse import csr_matrix

from misc import *
from Volc import *

# vocab is a list (index -> word mapping)
def runLDA(W, volc=None, nTopics=10, nIter=10, nTopicWords=1000, randomState=1, outfile=sys.stdout):
    if nTopicWords == -1:
        nTopicWords = len(vocab) # all words
    model = lda.LDA(n_topics=nTopics, n_iter=nIter, random_state=randomState)
    model.fit(W)
    if volc is not None:
        vocab = geti2W(volc)
        topicWord = model.topic_word_
        newVolc = Volc() # use top N words as volcabulary
        for i, topicDist in enumerate(topicWord):
            topicWords = list(np.array(vocab)[np.argsort(topicDist)][:-nTopicWords:-1])
            newVolc.addWord(toStr(topicWords))
        return model, volc
    else:
        return model

def geti2W(volc):
    if volc is None:
        return None
    i2w = [volc.getWord(i) for i in range(0, len(volc))]
    return i2w

# print Topic-Word Matrix [topicNum x wordNum] (phi in literature)
def printTWMatrix(model, i2w, encoding='utf-8', outfile=sys.stdout):
    for w in i2w:
        outfile.write((w + ',').encode(encoding))
    np.savetxt(outfile, model.topic_word_, delimiter=',')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:', sys.argv[0], 'pickle method top', file=sys.stderr)
        exit(-1)

