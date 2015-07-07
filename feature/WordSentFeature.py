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



# remove sentences not in allowedPOS set, and get volc
def preprocessDoc(lnList, volc, allowedPOS, df=None, minCnt=None):
    if volc is None:
        volc = Volc()
    
    for i, ln in enumerate(lnList):
        content = ln['news']['content_pos']
        sentList = list()
        for j, sent in enumerate(content.split(',')):
            hasWord = False
            for wt in sent.split(' '):
                (w, t) = wt.split('/')
                if w not in volc and volc.lockVolc: continue
                if allowedPOS != None and t not in allowedPOS: continue
                if df is not None and minCnt is not None and df[w] < minCnt: continue
                if w not in volc: volc.addWord(w)
                hasWord = True
            if hasWord:
               sentList.append(sent)
        ln['news']['content_pos'] = getContent(sentList)
    return lnList, volc
               
def getDF(lnList, volc, allowedPOS):
    if volc is None:
        volc = Volc()
    
    df = defaultdict(int)
    for i, ln in enumerate(lnList):
        content = ln['news']['content_pos']
        wordSet = set()
        for j, sent in enumerate(content.split(',')):
            for wt in sent.split(' '):
                (w, t) = wt.split('/')
                if w not in volc and volc.lockVolc: continue
                if allowedPOS != None and t not in allowedPOS: continue
                wordSet.add(w)
        for w in wordSet:
            df[w] += 1
    return df

def getContent(sentList):
    assert len(sentList) > 0
    content = ''
    for i, sent in enumerate(sentList):
        if i == 0:
            content = sent
        else:
            content = content + ',' + sent
    return content

class WordModel:

    def genPolarityX(self, lnList, volc, feature='tf'):
        numDoc = len(lnList)
        # sentence term frequency
        sentTF = [list() for i in range(0, numDoc)]  # sentTF[i][j]: tf of sententce j in doc i
        # document term frequency
        docTF = [defaultdict(int) for i in range(0, numDoc)]
        # document frequency
        df = defaultdict(int)
        for i, ln in enumerate(lnList):
            content = ln['news']['content_pos']
            wordSet = set()
            for j, sent in enumerate(content.split(',')):
                stf = defaultdict(int)
                for wt in sent.split(' '):
                    (w, t) = wt.split('/')
                    if w not in volc: continue
                    docTF[i][w] += 1
                    stf[w] += 1
                    wordSet.add(w)
                if len(stf) != 0:
                    sentTF[i].append(stf)
                else:
                    print('doc %d sent %d has no word: %s' % (i, j, sent), file=sys.stderr)
                    pass
            for w in wordSet:
                df[w] += 1
            if (i+1) % 100 == 0:
                print('%cProgress:(%d/%d)' % (13, i+1, len(lnList)), end='', file=sys.stderr)
        print('', file=sys.stderr)

        if feature == 'tfidf':
            # get idf
            idf = getIDF(df, numDoc)
            # calc tfidf
            for tf in docTF:
                for w in tf.keys():
                    tf[w] = tf[w] * idf[w]
            for tf in sentTF:
                for stf in tf:
                    for w in stf.keys():
                        stf[w] = stf[w] * idf[w]

        # convert to docX 
        rows, cols, data = list(), list(), list()
        nowRow = 0
        for i, tf in enumerate(docTF):
            for w, v in tf.items():
                if w in volc:
                    rows.append(i)
                    cols.append(volc[w])
                    data.append(v)
                else:
                    print('%s not in volc' % (w), file=sys.stderr) 
        dtype = np.int32 if feature == 'tf' else np.float64
        docX = csr_matrix((data, (rows, cols)), shape=(numDoc, len(volc)), dtype=dtype)

        # convert to sentX 
        rows, cols, data = list(), list(), list()
        nowRow = 0
        for i, tf in enumerate(sentTF):
            for j, stf in enumerate(tf):
                for w, v in stf.items():
                    if w in volc:
                        rows.append(nowRow)
                        cols.append(volc[w])
                        data.append(v)
                    else:
                        print('%s not in volc' % (w), file=sys.stderr) 
                nowRow += 1
        dtype = np.int32 if feature == 'tf' else np.float64
        sentX = csr_matrix((data, (rows, cols)), shape=(nowRow, len(volc)), dtype=dtype)
        
        # the mapping of doc to (the list of x's indexes)
        doc2XList = [list() for i in range(0, numDoc)]
        nowRow = 0
        for i, tf in enumerate(sentTF):
            for j, stf in enumerate(tf):
                doc2XList[i].append(nowRow)
                nowRow += 1
        return docX, sentX, doc2XList


    def genSubjectiveX(self, lnList, sVolc, feature='tf'):
        numDoc = len(lnList)
        # sentence term frequency
        sentTF = [list() for i in range(0, numDoc)]  # sentTF[i][j]: tf of sententce j in doc i
        # document frequency
        df = defaultdict(int)
        for i, ln in enumerate(lnList):
            content = ln['news']['content_pos']
            wordSet = set()
            for j, sent in enumerate(content.split(',')):
                stf = defaultdict(int)
                for wt in sent.split(' '):
                    (w, t) = wt.split('/')
                    if w not in sVolc: continue
                    stf[w] += 1
                    wordSet.add(w)
                sentTF[i].append(stf)
            for w in wordSet:
                df[w] += 1
            if (i+1) % 100 == 0:
                print('%cProgress:(%d/%d)' % (13, i+1, len(lnList)), end='', file=sys.stderr)
        print('', file=sys.stderr)

        if feature == 'tfidf':
            # get idf
            idf = getIDF(df, numDoc)
            # calc tfidf
            for tf in sentTF:
                for stf in tf:
                    for w in stf.keys():
                        stf[w] = stf[w] * idf[w]

        # convert to sentX 
        rows, cols, data = list(), list(), list()
        nowRow = 0
        for i, tf in enumerate(sentTF):
            for j, stf in enumerate(tf):
                for w, v in stf.items():
                    if w in sVolc:
                        rows.append(nowRow)
                        cols.append(sVolc[w])
                        data.append(v)
                    else:
                        print('%s not in sVolc' % (w), file=sys.stderr) 
                nowRow += 1
        dtype = np.int32 if feature == 'tf' else np.float64
        sentX = csr_matrix((data, (rows, cols)), shape=(nowRow, len(sVolc)), dtype=dtype)
        
        # the mapping of doc to (the list of x's indexes)
        doc2XList = [list() for i in range(0, numDoc)]
        nowRow = 0
        for i, tf in enumerate(sentTF):
            for j, stf in enumerate(tf):
                doc2XList[i].append(nowRow)
                nowRow += 1
        return sentX, doc2XList

# convert original list of doc ids to the list of sentences
def convert2SentIds(docIds, doc2XList):
    sentIds = list()
    for id in docIds:
        sentIds.extend(doc2XList[id])
    return sentIds

def expandLabel(labels, doc2XList):
    newLabels = list()
    for i, label in enumerate(labels):
        for j in range(0, len(doc2XList[i])):
            newLabels.append(label)
    return newLabels

def checkDoc2XList(l1, l2):
    for i, indexList1 in enumerate(l1):
        indexList2 = l2[i]
        assert len(indexList1) == len(indexList2)
        for j in range(0, len(indexList1)):
            assert indexList1[j] == indexList2[j]

# calculate subjective score for each sentence
def calcSubjectiveScore(sentSX):
    (sentNum, fNum) = sentX.shape
    ones = np.ones((fNum, 1))
    scoreList = [sentSX.getrow(i) * ones for i in range(0, sentNum)]
    return scoreList

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage:', sys.argv[0], 'TaggedLabelNewsJson modelConfigFile subjectiveLexion [minCnt]', file=sys.stderr)
        exit(-1)
    
    # arguments
    segLabelNewsJson = sys.argv[1]
    modelConfigFile = sys.argv[2]
    subjectiveWordFile = sys.argv[3]
    minCnt = None
    if len(sys.argv) == 5:
        minCnt = int(sys.argv[4])

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
    sVolc = Volc()
    sVolc.load(subjectiveWordFile)
    sVolc.lock()

    # parameters:
    fName = config['featureName']
    paramsIter = ParameterGrid(config['params'])

    wm = WordModel()

    # ============= Run for self-train-test ===============
    for t, lnList in sorted(lnListInTopic.items()):
        for p in paramsIter:
            (labelDocIds, unLabelDocIds) = getLabelIndex(lnList)
            labelLnList = [lnList[i] for i in labelDocIds]
            df = getDF(labelLnList, topicVolcDict[t], p['allowedPOS'])
            # preprocess 
            (lnList, volc) = preprocessDoc(lnList, topicVolcDict[t], p['allowedPOS'], df=df, minCnt=minCnt)

            ### generating polarity features ###
            # doc2XList[i]: the list of the sentX's index of doc i
            allDocPX, allSentPX, doc2XList = wm.genPolarityX(lnList, volc=volc, feature=p['feature'])
                        
            ### generating subjective features ###
            allSentSX, doc2XList2 = wm.genSubjectiveX(lnList, sVolc=sVolc, feature=p['feature'])

            checkDoc2XList(doc2XList, doc2XList2)

            (labelDocIds, unLabelDocIds) = getLabelIndex(lnList)
            labelSentIds = convert2SentIds(labelDocIds, doc2XList)
            unLabelSentIds = convert2SentIds(unLabelDocIds, doc2XList)
            
            # for splitting polarity document X
            docPX, unDocPX = allDocPX[labelDocIds], allDocPX[unLabelDocIds]

            # for splitting polarity sentence X, y
            sentPX, unSentPX = allSentPX[labelSentIds], allSentPX[unLabelSentIds]
            
            # for splitting subjective X
            sentSX, unSentSX = allSentSX[labelSentIds], allSentSX[unLabelSentIds]

            # for generating doc y and sentence y
            allDocy = np.array(getLabels(lnList))
            allSenty = np.array(expandLabel(getLabels(lnList), doc2XList))
            docy = allDocy[labelDocIds]
            senty = allSenty[labelSentIds]
            
            pObj = { 
                     'docy': docy, 'senty': senty, 
                     'docPX':docPX, 'unDocPX': unDocPX, 'sentPX': sentPX, 'unSentPX': unSentPX, 
                     'sentSX': sentSX, 'unSentSX': unSentSX, 'doc2XList': doc2XList,
                     'mainVolc': volc, 'config': config 
                }
            
            print('docy:', docy.shape, 'senty:', senty.shape, 'docPX:', docPX.shape, 
                    'unDocPX:', unDocPX.shape, 'sentPX:', sentPX.shape, 'unSentPX:', unSentPX.shape, 
                    'sentSX:', sentSX.shape, 'unSentSX:', unSentSX.shape)


            with open('t%d_%s_%s.pickle' % (t, fName, p['feature']),'w+b') as f:
                pickle.dump(pObj, f)



