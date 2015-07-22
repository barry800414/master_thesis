#!/usr/bin/env python3

import sys, json, math, pickle
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from misc import *
from Volc import *

'''
Generating BiWord(or even tri-word) feature (with shared volcabulary for running multi-task learning)
'''


def getVolc(lnList, allowedPOS, minCnt, volc=None):
    newVolc = Volc() if volc is None else volc.copy(lock=False)
    df = defaultdict(int)
    for i, ln in enumerate(lnList):
        content = ln['news']['content_in_tag_pos']
        wordSet = set()
        for j, sent in enumerate(content.split(',')):
            for wt in sent.split(' '):
                r = wt.split('/')
                if len(r) != 2:
                    print(r)
                    continue
                (w, t) = r
                if allowedPOS is not None and t not in allowedPOS: continue
                wordSet.add(w)
        
        content = ln['news']['content_out_tag_pos']
        for j, sent in enumerate(content.split(',')):
            for wt in sent.split(' '):
                r = wt.split('/')
                if len(r) != 2:
                    print(r)
                    continue
                (w, t) = r

                if allowedPOS is not None and t not in allowedPOS: continue
                wordSet.add(w)

        for w in wordSet:
            df[w] += 1
    
    for w, cnt in df.items():
        if cnt >= minCnt:
            newVolc.addWord(w)
    return newVolc


# scale: the scale of feature counting in in-tag text
def genX(lnList, allowedPOS, volc, inScale=2, outScale=1):
    if volc is None:
        volc = Volc()
    TF = list()
    for ln in lnList:
        sentList = ln['news']['content_in_tag_pos'].split(',')
        tf = defaultdict(int)
        for sent in sentList:
            for wt in sent.split(' '):
                r = wt.split('/')
                if len(r) != 2:
                    print(r)
                    continue
                (w, t) = r
                if allowedPOS is not None and t not in allowedPOS: continue
                if not volc.lockVolc and w not in volc:
                    volc.addWord(w)
                tf[w] += 1 * inScale
        
        sentList = ln['news']['content_out_tag_pos'].split(',')
        for sent in sentList:
            for wt in sent.split(' '):
                r = wt.split('/')
                if len(r) != 2:
                    print(r)
                    continue
                (w, t) = r
                if allowedPOS is not None and t not in allowedPOS: continue
                if not volc.lockVolc and w not in volc:
                    volc.addWord(w)
                tf[w] += 1 * outScale
        TF.append(tf)

    # generate X
    rows, cols, data = list(), list(), list()
    for i, tf in enumerate(TF):
        for w, cnt in tf.items():
            if w in volc:
                rows.append(i)
                cols.append(volc[w])
                data.append(cnt)
    dtype = np.float64 if type(inScale) == float else np.int32
    X = csr_matrix((data, (rows, cols)), shape=(len(TF), len(volc)), dtype=dtype)
    return X

    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:', sys.argv[0], 'TaggedLabelNewsJson inScale outScale', file=sys.stderr)
        exit(-1)
    
    lnListFile = sys.argv[1]
    inScale = float(sys.argv[2])
    outScale = float(sys.argv[3])

    with open(lnListFile, 'r') as f:
        lnList = json.load(f)

    lnListInTopic = divideLabelNewsByTopic(lnList)
    allowedPOS = set(['VV', 'VA', 'JJ', 'AD', 'NN', 'NR'])
    minCnt = 2

    # get the shared volcabulary first
    
    for t, lnList in lnListInTopic.items():
        volc = getVolc(lnList, allowedPOS, minCnt)
        allX = genX(lnList, allowedPOS, volc, inScale=inScale, outScale=outScale)
        ally = np.array(getLabels(lnList))
        (labelIndex, unLabelIndex) = getLabelIndex(lnList)
        X = allX[labelIndex]
        y = ally[labelIndex]
        unX = allX[unLabelIndex]

        pObj = { 'X':X, 'unX': unX, 'y':y, 'mainVolc': volc }
        with open('t%d_WeightedWord_in%g_out%g.pickle' % (t, inScale, outScale),'w+b') as f:
            pickle.dump(pObj, f)
        print('X:', X.shape)

