
import sys, json, pickle
import numpy as np
from collections import defaultdict

from Volc import Volc
from misc import *

# docLen: docLen / maxDocLen
# groupCnt: count of words in a group * (docLen / avgDocLen) => min max
# nUniqueWord: #unique words * (docLen / avgDocLen) => min max
# wordDiv: std of word distribution => min max
def genX(lnList, fType, params):
    lenList = [getDocLength(ln) for ln in lnList]
    avgLen = np.mean(lenList)
    scaleList = [len / avgLen for len in lenList]
    outVolc = Volc()
    if fType == 'docLen': # docLen/maxLen
        maxLen = max(lenList)
        lenRatioList = [float(len) / maxLen for len in lenList]
        X = np.array(lenRatioList, dtype=np.float64).reshape((-1, 1))
        outVolc.addWord('docLen')
    elif fType == 'groupCnt':
        # params should be a volc
        wordCntList = list()
        for i, ln in enumerate(lnList):
            wordCntList.append(getWordGroupCount(ln, params, scaleList[i]))
        X = np.array(wordCntList, dtype=np.float64) 
        for i in range(0, X.shape[1]):
            X[:,i] = X[:,i] / max(X[:,i])
        outVolc = params
    elif fType == 'nUniqueWord':
        # params should be a volc
        nUniqueWord= list()
        for i, ln in enumerate(lnList):
            nUniqueWord.append(getNUniqueWord(ln, params, scaleList[i]))
        X = np.array(nUniqueWord, dtype=np.float64).reshape((-1, 1))
        X = X / np.max(X)
        outVolc.addWord('nUniqueWord')
    elif fType == 'wordDiv': 
        # params should be a volc
        wordDiv = list()
        for ln in lnList:
            wordDiv.append(getWordDiversity(ln, params))
        X = np.array(wordDiv, dtype=np.float64).reshape((-1, 1))
        X = X / np.max(X)
        outVolc.addWord('wordDiv')

    return (X, outVolc)

def getVolc(lnList, minCnt=2):
    df = defaultdict(int)
    for ln in lnList:
        text = ln['news']['content_pos']
        wordSet = set()
        for sent in text.split(','):
            for wt in sent.split(' '):
                w, t = wt.split('/')
                wordSet.add(w)
        for w in wordSet:
            df[w] += 1
    volc = Volc()
    for w, cnt in df.items():
        if cnt >= minCnt:
            volc.addWord(w)
    return volc

# count of official/oppose entity
# if count
# sentiment dictionary count
def getWordGroupCount(ln, volc, scale=1):
    wordCnt = [0 for i in range(0, len(volc))]
    text = ln['news']['content_pos']
    for sent in text.split(','):
        for wt in sent.split(' '):
            w, t = wt.split('/')
            if w in volc:
                wordCnt[volc[w]] += 1
    return np.array(wordCnt, dtype=np.float64) * scale

# number of unique words in article, volc should be given
def getWordDiversity(ln, volc, scale=1):
    wordCnt = defaultdict(int)
    text = ln['news']['content_pos']
    for sent in text.split(','):
        for wt in sent.split(' '):
            w, t = wt.split('/')
            if w in volc:
                wordCnt[w] += 1
    return np.std(list(wordCnt.values()))

# number of unique words in article, volc should be given
def getNUniqueWord(ln, volc, scale=1):
    wordSet = set()
    text = ln['news']['content_pos']
    for sent in text.split(','):
        for wt in sent.split(' '):
            w, t = wt.split('/')
            if w in volc:
                wordSet.add(w)
    return len(wordSet) * scale

def getAvgDocLength(lnList):
    length = 0
    for ln in lnList:
        length += getDocLength(ln)
    return float(length) / len(lnList)

# article length 
def getDocLength(ln, scale=1):
    text = ln['news']['content_pos']
    length = 0
    for sent in text.split(','):
        length += len(sent.split(' '))
    return length * scale

if __name__ == '__main__':
    if len(sys.argv) < 5 :
        print('Usage:', sys.argv[0], 'taggedLabelNewsJsonFile minCnt type(docLen/groupCnt/nUniqueWord/wordDiv) outPickleSuffix [volc]', file=sys.stderr)
        exit(-1)

    lnListJsonFile = sys.argv[1]
    minCnt = int(sys.argv[2])
    fType = sys.argv[3]
    outFileSuffix = sys.argv[4]
    volcFile = sys.argv[5] if len(sys.argv) == 6 else None

    # load label news 
    with open(lnListJsonFile, 'r') as f:
        labelNewsList = json.load(f)

    inVolc = None
    if volcFile is not None:
        inVolc = Volc()
        inVolc.load(volcFile)

    # get the set of all possible topic
    lnListInTopic = divideLabelNewsByTopic(labelNewsList)
    
    for t, lnList in sorted(lnListInTopic.items()):
        (labelIndex, unLabelIndex) = getLabelIndex(lnList)
        labelLnList = [lnList[i] for i in labelIndex]
        volc = getVolc(labelLnList, minCnt=minCnt)
        if fType in ['docLen', 'nUniqueWord', 'wordDiv'] :
            X, outVolc = genX(labelLnList, fType, volc)
        elif fType == 'groupCnt':
            X, outVolc = genX(labelLnList, fType, inVolc)

        ally = np.array(getLabels(lnList))
        y = ally[labelIndex]
        
        print(X.shape, X.transpose())

        pObj = { 'X':X, 'unX': None, 'y':y, 'mainVolc': outVolc, 'config': fType }
        with open('t%d_%s.pickle' % (t, outFileSuffix),'w+b') as f:
            pickle.dump(pObj, f)



