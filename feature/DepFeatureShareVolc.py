#!/usr/bin/env python3

import sys, json, pickle
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.grid_search import ParameterGrid

import TreePattern as TP
import DepTree as DT
from PhraseDepTree import *
import NegPattern as NP
from Volc import *
from Opinion import *
from misc import *

class OpinionModel:
    def __init__(self, depParsedLabelNews, topicPhraseList=None):
        self.setModelFlag = False
        self.pln = depParsedLabelNews
        self.tPL = topicPhraseList
        self.init()

    # generate (phrase) dependency trees
    def init(self):
        print('Generating dependency trees ... ', end='',file=sys.stderr)
        corpusDTList = list()
        for i, ln in enumerate(self.pln):
            newsDTList = list()
            topicId = ln['statement_id']
            contentDep = ln['news']['content_dep']
            # for each dependency tree
            for depObj in contentDep:
                tdList = depObj['tdList']
                if self.tPL != None:
                    depTree = PDT.PhraseDepTree(tdList, self.tPL[topicId])
                else:
                    depTree = DT.DepTree(tdList)
                newsDTList.append(depTree)
            corpusDTList.append(newsDTList)
            if (i+1) % 100 == 0:
                print('%cGenerating dependency trees ... Progress:(%d/%d)' % (13, i+1, len(self.pln)), end='', file=sys.stderr)
        print('', file=sys.stderr)
        self.corpusDTList = corpusDTList


    def getVolcDict(self):
        return self.volcDict

    # copy original volcabulary
    def setVolcDict(self, volcDict):
        if volcDict is None:
            self.volcDict = { 'main': Volc(), 'holder': None, 'opinion': None, 'target': None }
        else:
            self.volcDict = dict()
            for type in ['main', 'holder', 'opinion', 'target']:
                if type in volcDict and volcDict[type] is not None:
                    self.volcDict[type] = volcDict[type].copy(lock=True)
                else:
                    self.volcDict[type] = None
            if self.volcDict['main'] is None:
                self.volcDict['main'] = Volc()

    # keyTypeList: The list of key types to be used as opinion key ('HOT', 'HT', 'OT', 'HO', 'T', 'H')
    # opnNameList: The list of selected opinion name 
    # sentiDict: sentiment dictionary
    # negSepList: the list of boolean flag to indicate whether negation pattern is represented separated
    def getVolcFromData(self, pTreeList, negPList, sentiDict, keyTypeList, 
            opnNameList, negSepList, ignoreNeutral, pTreeSepList, countTreeMatched, 
            minCnt=2, volc=None):
        self.pTL = pTreeList
        self.nPL = negPList
        self.kTL = keyTypeList
        self.opnNL = opnNameList
        self.sD = sentiDict
        self.nSL = negSepList
        self.iN = ignoreNeutral
        self.pTSL = pTreeSepList
        self.cTM = countTreeMatched # count the number of matched of that tree pattern
        self.minCnt = minCnt 
        self.setVolcDict(None)

        print('Extracting Opinions(tree pattern matching) ...', end='',file=sys.stderr)
        docOpnCnt = defaultdict(int) # opnKey -> document frequency
        for i, newsDTList in enumerate(self.corpusDTList):
            opnDict = self.extractOpn(newsDTList)
            opnCnt = self.countOpn(opnDict)
            for key in opnCnt.keys():
                docOpnCnt[key] += 1
            
            if (i+1) % 100 == 0:
                print('%cExtracting Opinions(tree pattern matching) ... Progress(%d/%d)' % (13, i+1, 
                    len(self.corpusDTList)), end='', file=sys.stderr) 
        print('', file=sys.stderr)
        
        # get volcabulary
        volc = Volc() if volc is None else volc
        for key, cnt in docOpnCnt.items():
            if cnt >= minCnt:
                volc.addWord(key)        
        
        return volc

    # keyTypeList: The list of key types to be used as opinion key ('HOT', 'HT', 'OT', 'HO', 'T', 'H')
    # opnNameList: The list of selected opinion name 
    # sentiDict: sentiment dictionary
    # negSepList: the list of boolean flag to indicate whether negation pattern is represented separated
    def genX(self, pTreeList, negPList, sentiDict, volcDict, keyTypeList, 
            opnNameList, negSepList, ignoreNeutral, pTreeSepList, countTreeMatched):
        self.pTL = pTreeList
        self.nPL = negPList
        self.kTL = keyTypeList
        self.opnNL = opnNameList
        self.sD = sentiDict
        self.nSL = negSepList
        self.iN = ignoreNeutral
        self.pTSL = pTreeSepList
        self.cTM = countTreeMatched # count the number of matched of that tree pattern
        self.setVolcDict(volcDict)

        print('Extracting Opinions(tree pattern matching) ...', end='',file=sys.stderr)
        opnCntList = list() # the list to save all opinions in each document
        for i, newsDTList in enumerate(self.corpusDTList):
            opnDict = self.extractOpn(newsDTList)
            opnCnt = self.countOpn(opnDict, volc)
            opnCntList.append(opnCnt)
            if (i+1) % 100 == 0:
                print('%cExtracting Opinions(tree pattern matching) ... Progress(%d/%d)' % (13, i+1, 
                    len(self.corpusDTList)), end='', file=sys.stderr) 
        print('', file=sys.stderr)

        # convert to X, y
        rows = list()
        cols = list()
        entries = list()
        for rowId, opnCnt in enumerate(opnCntList):
            for opnKey, value in opnCnt.items():
                if opnKey not in volc:
                    continue
                colId = volc[opnKey]
                rows.append(rowId)
                cols.append(colId)
                entries.append(value)
        numRow = len(opnCntList)
        numCol = len(volc)
        X = csr_matrix((entries, (rows, cols)), shape=(numRow, 
            numCol), dtype=np.int32)
        
        return X

    # opnDict: a dictionary (opinion-type-name -> list of opinions)
    # return: a dictionary (opnKey -> count) 
    def countOpn(self, opnDict, volc=None):
        opnCnt = defaultdict(int)
        for opnName, opns in opnDict.items():
            # ignore the opinions which are not selected
            if self.opnNL is not None and opnName not in self.opnNL:
                continue
            for opn in opns:
                for keyType in self.kTL:
                    for negSep in self.nSL:
                        for pTreeSep in self.pTSL:
                            keyValue = OpinionModel.getOpnKeyValue(opn, keyType, self.sD, negSep, self.iN, pTreeSep)
                            if keyValue is not None:
                                (key, value) = keyValue
                                if volc is not None and key not in volc:
                                    continue
                                opnCnt[key] += value

            # if countTreeMatched is on
            if self.cTM:
                if volc is not None and opnName not in volc:
                    continue
                opnCnt[opnName] = len(opns)
                        
        return opnCnt
        

    # depParsedNews: dependency parsed news 
    # return: a dictionary (opinion-type-name -> list of opinions)
    def extractOpn(self, newsDTList):
        opnDict = dict()
        
        # for each dependency tree
        for depTree in newsDTList:
            # for each pattern tree
            for pTree in self.pTL:
                if pTree.name not in opnDict:
                    opnDict[pTree.name] = list()
                results = pTree.match(depTree) # a list of opinions (dict)
                
                # find negation pattern
                if self.nPL != None:
                    for r in results:
                        negCntDict = NP.checkAllNegPattern(self.nPL,
                                depTree, pTree, r['mapping'])
                        if len(negCntDict) > 0:
                            r['neg'] = negCntDict
                        del r['mapping']
                
                # convert to Opinion objects
                opnList = list()
                for i in range(0, len(results)):
                    opn = Opinion.genOpnFromDict(results[i], self.volcDict, pTree.name)
                    if opn is None:
                        continue
                    opnList.append(opn)
                opnDict[pTree.name].extend(opnList)
        return opnDict

    # get key of opinion object
    def getOpnKeyValue(opn, keyType, sentiDict=None, negSep=False, ignoreNeutral=False, pTreeSep=False):
        if keyType == 'HT' or keyType == 'T' or keyType == 'H':
            assert sentiDict != None
        
        if keyType == 'HOT':
            return opn.getKeyHOT(negSep, pTreeSep)
        elif keyType == 'HT':
            return opn.getKeyHT(sentiDict, negSep, ignoreNeutral, pTreeSep)
        elif keyType == 'H':
            return opn.getKeyH(sentiDict, negSep, ignoreNeutral, pTreeSep)
        elif keyType == 'HO':
            return opn.getKeyHO(negSep, pTreeSep)
        elif keyType == 'OT':
            return opn.getKeyOT(negSep, pTreeSep)
        elif keyType == 'T':
            return opn.getKeyT(sentiDict, negSep, ignoreNeutral, pTreeSep)

    def printOpnCnt(opnCnts, outfile=sys.stdout):
        for opnName, opnCnt in opnCnts.items():
            print(opnName, file=outfile)
            for key, cnt in sorted(opnCnt.items(), key = lambda x:x[1], reverse=True):
                print(key, cnt, sep=',', file=outfile)

    # assume all are negative Separated
    def addNewPropOpnKey(self, newOpnCnt, cnt, adjList, opnKey, opnKeySet):
        keyType = opnKey[0]
        if keyType == 'HOT':
            (i1, i2, i3) = (opnKey[1], opnKey[2], opnKey[4])
            for newi1, p1, in adjList[i1]:
                for newi2, p2 in adjList[i2]:
                    for newi3, p3 in adjList[i3]:
                        newKey = (opnKey[0], newi1, newi2, opnKey[3], newi3)
                        if newKey in opnKeySet:
                            newOpnCnt[newKey] += p1 * p2 * p3 * cnt
        elif keyType in ['HO', 'OT', 'HT']:
            (i1, i2) = (opnKey[1], opnKey[3])
            for newi1, p1, in adjList[i1]:
                for newi2, p2 in adjList[i2]:
                    newKey = (opnKey[0], newi1, opnKey[2], newi2)
                    if newKey in opnKeySet:
                        newOpnCnt[newKey] += p1 * p2 * cnt
        elif keyType in ['T', 'H']:
            i1 = opnKey[1]
            for newi1, p1 in adjList[i1]:
                newKey = (opnKey[0], newi1, opnKey[2])
                if newKey in opnKeySet:
                    newOpnCnt[newKey] += p1 * cnt


def initOM(lnList, topicPhraseList=None):
    om = OpinionModel(lnList, topicPhraseList)
    return om

# generate opinion model features
def genXY(om, params, preprocess, minCnt, pTreeList, negPList=None, sentiDict=None, volcDict=None, wordGraph=None, wgParams=None):
    print('generating opinion model features ...', file=sys.stderr)
    p = params
    (X, y) = om.genXY(pTreeList, negPList, sentiDict, volcDict, p['keyTypeList'], 
            p['opnNameList'], p['negSepList'], p['ignoreNeutral'], p['pTreeSepList'], 
            p['countTreeMatched'],  minCnt, wordGraph, wgParams)
    if preprocess != None:
        X = DataTool.preprocessX(X, preprocess['method'], preprocess['params'])
    volcDict = om.getVolcDict()
    assert X.shape[1] == len(volcDict['main'])
    return (X, y, volcDict)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage:', sys.argv[0], 'DepParsedLabelNews ModelConfigFile NegPatternFile SentiDictFile', file=sys.stderr)
        exit(-1)

    depParsedLabelNewsJsonFile = sys.argv[1]
    modelConfigFile = sys.argv[2]
    negPatternFile = sys.argv[3]
    sentiDictFile = sys.argv[4]

    # load model config
    with open(modelConfigFile, 'r') as f:
        config = json.load(f)
    # load label news
    with open(depParsedLabelNewsJsonFile, 'r') as f:
        lnList = json.load(f)
    
    # get the set of all possible topic
    topicSet = set([ln['statement_id'] for ln in lnList])
    lnListInTopic = divideLabelNewsByTopic(lnList)

    # load pattern trees 
    pTreeList = TP.loadPatterns(config['treePattern'])
    # load negation pattern file
    negPList = NP.loadNegPatterns(negPatternFile)
    # load sentiment dictionary
    sentiDict = readSentiDict(sentiDictFile)
    # load volcabulary file
    topicVolcDict = loadVolcFileFromConfig(config['volc'], topicSet)
    # load phrase file
    topicPhraseList = loadPhraseFileFromConfig(config['phrase'])

    # model parameters 
    fName = config['featureName']
    paramsIter = ParameterGrid(config['params'])

    # init
    
    # generate volcabulary from all topic's labeled data first
    volc = Volc()
    for t, lnList in lnListInTopic.items():
        if t == 2: continue
        (labelIndex, unLabelIndex) = getLabelIndex(lnList)
        labelLnList = [lnList[i] for i in labelIndex]
        om = initOM(labelLnList, topicPhraseList)
        for p in paramsIter:
            volc = om.getVolcFromData(pTreeList, negPList, sentiDict, p['keyTypeList'], 
                p['opnNameList'], p['negSepList'], p['ignoreNeutral'], p['pTreeSepList'], 
                p['countTreeMatched'], minCnt=2, volc=volc)
    
    om = { t: initOM(ln, topicPhraseList) for t, ln in lnListInTopic.items() if t != 2 }
    for t, lnList in sorted(lnListInTopic.items(), key=lambda x:x[0]):
        if t == 2: continue
        for p in paramsIter:
            allX = om[t].genX(pTreeList, negPList, sentiDict, { 'main': volc }, p['keyTypeList'], 
                p['opnNameList'], p['negSepList'], p['ignoreNeutral'], p['pTreeSepList'], p['countTreeMatched'])
            ally = np.array(getLabels(lnList))
            (labelIndex, unLabelIndex) = getLabelIndex(lnList)
            X = allX[labelIndex]
            y = ally[labelIndex]
            unX = allX[unLabelIndex]
            
            pObj = { 'X':X, 'unX': unX, 'y':y, 'mainVolc': volc, 'config': config }
            with open('t%d_%s_shareVolc.pickle' % (t, fName),'w+b') as f:
                pickle.dump(pObj, f)

            print('t:', t, 'XShape:', X.shape, 'volcSize:', len(volc), file=sys.stderr)
