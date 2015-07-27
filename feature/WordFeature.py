#!/usr/bin/env python3

import sys, json, math, pickle
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, hstack
from sklearn.grid_search import ParameterGrid

from misc import *
from Volc import *

'''
This is the improved version of WordModel
 1.remove < 1 dimension (DONE)
 2.allowed some of pos taggers (DONE)
 3.word clustering 
 4.highest tfidf

Author: Wei-Ming Chen
Date: 2015/05/04

'''

class WordModel:
    # get volcabulary, and document length 
    def preCalc(lnList, allowedPOS, minCnt, volc=None):
        newVolc = Volc() if volc is None else volc.copy(lock=False)
        df = defaultdict(int)
        docLenList = list()
        for i, ln in enumerate(lnList):
            content = ln['news']['content_pos']
            wordSet = set()
            docLen = 0
            for j, sent in enumerate(content.split(',')):
                wtList = sent.split(' ')
                docLen += len(wtList)
                for wt in wtList:
                    (w, t) = wt.split('/')
                    if allowedPOS is not None and t not in allowedPOS: continue
                    wordSet.add(w)
            for w in wordSet:
                df[w] += 1
            docLenList.append(docLen)
        for w, cnt in df.items():
            if cnt >= minCnt:
                newVolc.addWord(w)
        return newVolc, docLenList

    def genX(self, lnList, feature='tf', volcDict=None, allowedPOS=None, minCnt=None, scaleList=None):
        self.setVolcDict(volcDict)
        volc = self.volcDict['main']
        IDF = None
        zeroOne = False
        if feature in ['0/1', '01']:
            zeroOne = True

        # calculate document frequency & generate volcabulary in advance
        (DF, volc) = WordModel.calcDF(lnList, volc=volc)
        
        # calcualte IDF if necessary
        if feature in ['tfidf', 'tf-idf']:
            IDF = WordModel.DF2IDF(DF, len(lnList))
        
        # calculate TF/TF-IDF (content)
        newsTFIDF = None
        (newsTFIDF, volc) = WordModel.corpusToTFIDF(lnList,
                        allowedPOS=allowedPOS, IDF=IDF, volc=volc, 
                        zeroOne=zeroOne, scaleList=scaleList)
        
        # generate X
        dtype = np.int32 if feature == 'tf' else np.float64
        X = toMatrix(newsTFIDF, volc, matrixType='csr', dtype=dtype)

        # remove the words whose document frequency <= threshold
        if minCnt != None:
            print('Before:', len(volc), file=sys.stderr)
            DF = countDFByCSRMatrix(X)
            X = shrinkCSRMatrixByDF(X, DF, minCnt)
            DF = volc.shrinkVolcByDocF(DF, minCnt)
            print('After:', len(volc), file=sys.stderr)

        # update volc
        self.volcDict['main'] = volc

        return X

    # copy original volcabulary
    def setVolcDict(self, volcDict):
        if volcDict is None:
            self.volcDict = { 'main': None }
        else:
            self.volcDict = dict()
            self.volcDict['main'] = volcDict['main'].copy()

    def getVolcDict(self):
        return self.volcDict

    # convert the corpus of news to tf/tf-idf (a list of dict)
    def corpusToTFIDF(lnList, volc, allowedPOS=None, IDF=None, zeroOne=False, scaleList=None):
        vectorList = list() # a list of dict()
        for i, ln in enumerate(lnList):
            text = ln['news']['content_pos']
            scale = scaleList[i] if scaleList is not None else 1
            vectorList.append(WordModel.text2TFIDF(text, volc, 
                allowedPOS, IDF, zeroOne, scale))
        return (vectorList, volc)

    # convert text to TF-IDF features (dict)
    def text2TFIDF(text, volc, allowedPOS=None, IDF=None, zeroOne=False, 
            scale = 1, sentSep=",", wordSep=" ", tagSep='/'):
        f = dict()
        # term frequency 
        for sent in text.split(sentSep):
            for wt in sent.split(wordSep):
                (word, tag) = wt.split(tagSep)
                if allowedPOS != None and tag not in allowedPOS:
                    continue
                if word not in volc:
                    continue
                
                if volc[word] not in f:
                    f[volc[word]] = scale
                else:
                    if not zeroOne: #if not zeroOne, calculate the count
                        f[volc[word]] += scale
                    
        # if idf is given, then calculate tf-idf
        if IDF is not None:
            for key, value in f.items():
                if key not in IDF:
                    #print('Document Frequency Error', file=sys.stderr)
                    f[key] = value * IDF['default']
                else:
                    f[key] = value * IDF[key]
        else:
            for key in f.keys():
                f[key] = getRound(f[key])

        return f

    # calculate document frequency
    def calcDF(lnList, sentSep=",", wordSep=" ", tagSep='/', volc=None):
        if volc == None:
            volc = Volc()
        # calculating docuemnt frequency
        docF = defaultdict(int)
        for ln in lnList:
            wordIdSet = set()
            text = ln['news']['content_pos']
            for sent in text.split(sentSep):
                for wt in sent.split(wordSep):
                    #print(wt)
                    (word, tag) = wt.split(tagSep)
                    if word not in volc and not volc.lockVolc: # building volcabulary
                        volc.addWord(word)
                        wordIdSet.add(volc[word])

            for word in wordIdSet:
                docF[word] += 1
        return (docF, volc)

    # convert document frequency to inverse document frequency
    def DF2IDF(DF, docNum):
        # calculate IDF (log(N/(nd+1)))
        IDF = dict()
        for key, value in DF.items():
            IDF[key] = math.log(float(docNum+1) / (value + 1))
        IDF['default'] = math.log(float(docNum+1))
        return IDF



def toMatrix(listOfDict, volc, matrixType='csr', dtype=np.float64):
    rows = list()
    cols = list()
    entries = list()
    for rowId, dictObject in enumerate(listOfDict):
        for colId, value in dictObject.items():
            rows.append(rowId)
            cols.append(colId)
            entries.append(value)
    numRow = len(listOfDict)
    numCol = len(volc)
    if matrixType == 'csr':
        m = csr_matrix((entries, (rows, cols)), 
                shape=(numRow, numCol), dtype=dtype)
    elif matrixType == 'csc':
        m = csc_matrix((entries, (rows, cols)), 
                shape=(numRow, numCol), dtype=dtype)
    else:
        m = None
    return m
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'TaggedLabelNewsJson modelConfigFile', file=sys.stderr)
        exit(-1)
    
    # arguments
    segLabelNewsJson = sys.argv[1]
    modelConfigFile = sys.argv[2]

    # load label news 
    with open(segLabelNewsJson, 'r') as f:
        lnList = json.load(f)
    # load model config
    with open(modelConfigFile, 'r') as f:
        config = json.load(f)

    # get the set of all possible topic
    lnListInTopic = divideLabelNewsByTopic(lnList)
    topicSet = set(lnListInTopic.keys())

    # load volcabulary file
    topicVolcDict = loadVolcFileFromConfig(config['volc'], topicSet)

    # parameters:
    fName = config['featureName']
    paramsIter = ParameterGrid(config['params'])
    lengthNorm = config['lengthNorm'] if 'lengthNorm' in config else False

    wm = WordModel()

    # ============= Run for self-train-test ===============
    for t, lnList in sorted(lnListInTopic.items()):
        scaleList = None
        if lengthNorm:
            volc, docLenList = WordModel.preCalc(lnList, None, 0)
            #print(docLenList)
            scaleList = convert2ScaleList(docLenList)
        for p in paramsIter:
            # there could be unlabeled data
            allX = wm.genX(lnList, feature=p['feature'], volcDict=topicVolcDict[t], 
                    allowedPOS=p['allowedPOS'], scaleList=scaleList)
            volcDict = wm.getVolcDict()
            ally = np.array(getLabels(lnList))
            (labelIndex, unLabelIndex) = getLabelIndex(lnList)
            X = allX[labelIndex]
            y = ally[labelIndex]
            unX = allX[unLabelIndex]
            
            pObj = { 'X':X, 'unX': unX, 'y':y, 'mainVolc': volcDict['main'], 'config': config }
            print(X.shape, file=sys.stderr)
            #with open('t%d_%s_%s.pickle' % (t, fName, p['feature']),'w+b') as f:
            #    pickle.dump(pObj, f)



