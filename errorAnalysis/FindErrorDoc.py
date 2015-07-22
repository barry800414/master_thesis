
import sys, pickle, json
from collections import defaultdict
from misc import *
from ErrorAnalysis import *
from Color import *

def calcErrNewsCnt(logP, y, topic, testCnt, errCnt):
    for log in logP:
        clf = log['clf']
        testIndex = log['split']['testIndex']
        yTest = y[testIndex]
        yTestPredict = log['predict']['yTestPredict']
        assert len(yTest) == len(yTestPredict)
        for newI, oldI in enumerate(testIndex):
            if yTest[newI] != yTestPredict[newI]:
                errCnt[oldI] += 1
            testCnt[oldI] += 1
                
# read label mapping 
def readLabelMapping(filename):
    newMap = list()
    oldMap = list()
    with open(filename, 'r') as f:
        entry = f.readline().strip().split(' ')
        topic = int(entry[0])
        num = int(entry[1])
        for line in f:
            entry = line.strip().split(' ')
            newMap.append(entry[0])
            oldMap.append(entry[1])
        assert len(newMap) == num
    return topic, newMap, oldMap

# read label-news json file
def readLnList(filename):
    with open(filename, 'r') as f:
        lnList = json.load(f)
    topicLnList = divideLabelNewsByTopic(lnList)
    topicNewsId = { t: [ln['news_id'] for ln in lns] for t, lns in topicLnList.items() }
    return lnList, topicLnList, topicNewsId


def getWordColorMap(clf, volc, c1Color, c0Color, topN):
    coef = clf.coef_
    cNum = coef.shape[0] # classNum
    cList = clf.classes_
    
    fNum = coef.shape[1] # feature number
    cValues = list()
    values = [(i, v) for i, v in enumerate(coef[0])]
    values.sort(key=lambda x:x[1], reverse=True)
    
    c1WordSet = set([volc.getWord(i) for i,v in values[0:topN]])
    c0WordSet = set([volc.getWord(i) for i,v in values[-topN:]])

    wc = dict()
    for w in c1WordSet:
        wc[w] = c1Color
    for w in c0WordSet:
        wc[w] = c0Color
    return wc

def colorTaggedText(text, wordColorMap):
    outStr = ''
    for sent in text.split(','):
        for wt in sent.split(' '):
            (w, t) = wt.split('/')
            if w in wordColorMap:
                outStr += cm[wordColorMap[w]] + w + cm['no']  # cm: color mapping
            else:
                outStr += w
        outStr += ','
    return outStr

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage:', sys.argv[0], 'dataPickle labelMappingFile labelNewsJsonFile logPickle1 logPickle2 ...', file=sys.stderr)
        exit(-1)

    # load data pickle
    with open(sys.argv[1], 'r+b') as f:
        dataP = pickle.load(f)
    y = dataP['y']
    # read label mapping file
    topic, newMap, oldMap = readLabelMapping(sys.argv[2])
    # read label news json file
    lnList, topicLnList, topicNewsId = readLnList(sys.argv[3])

    errCnt = [0 for i in range(0, len(topicLnList[topic]))]
    testCnt = [0 for i in range(0, len(topicLnList[topic]))]

    for i in range(4, len(sys.argv)):
        filename = sys.argv[i]
        print('Loading pickle file:', filename, file=sys.stderr)
        with open(filename, 'r+b') as f:
            logP = pickle.load(f)
        calcErrNewsCnt(logP, y, topic, testCnt, errCnt)


    # prepare coloring contents
    clf = logP[0]['clf']
    volc = dataP['mainVolc']
    topN = 100
    wordColorMap = getWordColorMap(clf, volc, 'green', 'red', topN)
    
    errRate = [(i, errCnt[i]/testCnt[i]) for i in range(0, len(errCnt))]
    errRate.sort(key = lambda x:x[1], reverse=True)
    for i, rate in errRate:
        print('Error Rate: %f(%d/%d)' % (rate, errCnt[i], testCnt[i]), end='')
        print(' new:%s  old:%s' % (newMap[i], oldMap[i]))
        print('News:', colorTaggedText(topicLnList[topic][i]['news']['content_pos'], wordColorMap))
        print('-'*100)

    
