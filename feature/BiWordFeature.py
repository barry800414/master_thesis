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

# wSize: window size
def getVolc(lnList, wSize, allowedPOS, deniedPOS, minCnt, volc=None):
    df = defaultdict(int) # document frequency
    for ln in lnList:
        nWordSet = set()
        sentList = ln['news']['content_pos'].split(',')
        for sent in sentList:
            wtList = sent.split(' ')
            for i in range(0, len(wtList)):
                wtList[i] = wtList[i].split('/') # the list of (w, t) in sentence

            for i in range(0, len(wtList) - wSize + 1):
                nWord = wtList[i:i+wSize]
                select = False
                for w, t in nWord:
                    if t in allowedPOS:
                        select = True
                    if t in deniedPOS:
                        select = False
                        break
                if select:
                    nWord = tuple([w for w, t in nWord])
                    nWordSet.add(nWord)
        for nWord in nWordSet:
            df[nWord] += 1
                   
    print('Original:', len(df), end=' ', file=sys.stderr)
    volc = Volc() if volc is None else volc
    wvList = list()
    for nWord, cnt in df.items():
        if cnt >= minCnt:
            volc.addWord(nWord)
            wvList.append((nWord, cnt))
    print('Later:', len(volc), file=sys.stderr)
    wvList.sort(key=lambda x:x[1], reverse=True)
    #for nWord, cnt in wvList:
    #    print(nWord, cnt)

    return volc

def genX(lnList, wSize, allowedPOS, deniedPOS, volc):
    nWordCntList = list()
    for ln in lnList:
        sentList = ln['news']['content_pos'].split(',')
        nWordCnt = defaultdict(int)
        for sent in sentList:
            wtList = sent.split(' ')
            for i in range(0, len(wtList)):
                wtList[i] = wtList[i].split('/') # the list of (w, t) in sentence

            for i in range(0, len(wtList) - wSize + 1):
                nWord = wtList[i:i+wSize]
                select = False
                for w, t in nWord:
                    if t in allowedPOS:
                        select = True
                    if t in deniedPOS:
                        select = False
                        break
                if select:
                    nWord = tuple([w for w, t in nWord])
                    nWordCnt[nWord] += 1
        nWordCntList.append(nWordCnt)

    # generate X
    rows, cols, data = list(), list(), list()
    for i, nWordCnt in enumerate(nWordCntList):
        for nWord, cnt in nWordCnt.items():
            if nWord in volc:
                rows.append(i)
                cols.append(volc[nWord])
                data.append(cnt)
    X = csr_matrix((data, (rows, cols)), shape=(len(nWordCntList), len(volc)), dtype=np.int32)
    return X

    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:', sys.argv[0], 'TaggedLabelNewsJson windowSize [-config configFile] [-shareVolc]', file=sys.stderr)
        exit(-1)
    
    lnListFile = sys.argv[1]
    wSize = int(sys.argv[2])
    shareVolc = False
    config = None
    for i in range(3, len(sys.argv)):
        if sys.argv[i] == '-shareVolc':
            shareVolc = True
        elif sys.argv[i] == '-config' and len(sys.argv) > i:
            with open(sys.argv[i+1], 'r') as f:
                config = json.load(f)

    with open(lnListFile, 'r') as f:
        lnList = json.load(f)

    lnListInTopic = divideLabelNewsByTopic(lnList)
    topicSet = set(lnListInTopic.keys())
    allowedPOS = set(['VV', 'VA', 'JJ', 'AD', 'NN', 'NR'])
    deniedPOS = set(['PU'])
    minCnt = 0
    topicVolcDict = None

    if config is not None:
        # load volcabulary file
        topicVolcDict = loadVolcFileFromConfig(config['volc'], topicSet)

    if shareVolc:
        mainVolc = Volc() # as main volcabulary
        suffix = '%dWord_shareVolc' % wSize
        for t, lnList in lnListInTopic.items():
            if t == 2: continue
            (labelIndex, unLabelIndex) = getLabelIndex(lnList)
            labelLnList = [lnList[i] for i in labelIndex]
            volc = getVolc(labelLnList, wSize, allowedPOS, deniedPOS, minCnt, volc)
    else:
        suffix = '%dWord' % wSize

    for t, lnList in lnListInTopic.items():
        if t == 2: continue
        (labelIndex, unLabelIndex) = getLabelIndex(lnList)
        labelLnList = [lnList[i] for i in labelIndex]
        volc = volc if shareVolc else getVolc(labelLnList, wSize, allowedPOS, deniedPOS, minCnt)
        allX = genX(lnList, wSize, allowedPOS, deniedPOS, volc)
        ally = np.array(getLabels(lnList))
        X = allX[labelIndex]
        y = ally[labelIndex]
        unX = allX[unLabelIndex]
        print('t:', t, 'Xshape:', X.shape, file=sys.stderr)
        pObj = { 'X':X, 'unX': unX, 'y':y, 'mainVolc': volc }
        with open('t%d_%s.pickle' % (t, suffix),'w+b') as f:
            pickle.dump(pObj, f)

