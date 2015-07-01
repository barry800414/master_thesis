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
def runLDA(W, vocab=None, nTopics=10, nIter=10, nTopicWords=10, randomState=1, outfile=sys.stdout):
    if nTopicWords == -1:
        nTopicWords = len(vocab) # all words
    model = lda.LDA(n_topics=nTopics, n_iter=nIter, random_state=randomState)
    model.fit(W)
    if vocab is not None:
        topicWord = model.topic_word_
        topicWordList = list()
        for i, topicDist in enumerate(topicWord):
            topicWords = list(np.array(vocab)[np.argsort(topicDist)][:-nTopicWords:-1])
            topicWordList.append(topicWords)
            print('Topic {}: {}'.format(i, ' '.join(topicWords)), file=outfile)
    return model

# print Topic-Word Matrix [topicNum x wordNum] (phi in literature)
def printTWMatrix(model, i2w, encoding='utf-8', outfile=sys.stdout):
    for w in i2w:
        outfile.write((w + ',').encode(encoding))
    np.savetxt(outfile, model.topic_word_, delimiter=',')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:', sys.argv[0], 'pickle method top', file=sys.stderr)
        exit(-1)

