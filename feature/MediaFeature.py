
import sys, json, pickle

from scipy.sparse import csr_matrix
import numpy as np
from sklearn.grid_search import ParameterGrid

from Volc import Volc
from RunExperiments import *
from misc import *

def genXY(lnList, mediaVolc):
    rows = list()
    cols = list()
    data = list()
    for i, ln in enumerate(lnList):
        media = getMedia(ln['news_id'])
        mId = mediaVolc[media]
        rows.append(i)
        cols.append(mId)
        data.append(1)
    X = csr_matrix((data, (rows, cols)), shape=(len(lnList), len(mediaVolc)))
    y = np.array(getLabels(lnList))
    return (X, y)

def getMediaVolc(lnList):
    mediaVolc = Volc()
    for topicId, lnList in lnInTopic.items():
        for ln in lnList:
            media = getMedia(ln['news_id'])
            mediaVolc.addWord(media)
    return mediaVolc

def printMediaStat(lnInTopic):
    mediaName = { '02_coolloud':"苦勞", '07_setn':"三立", '05_ltn':'自由', '03_stormmedia':"風傳", '06_udn':'聯合', '01_pnn':'公視', '04_thenewslens':'關鍵' }
    statName = { 2: "台灣不應停建核四", 3: "應簽訂服務貿易協議", 4: "應簽訂自由經濟示範區條例", 5: "台灣應進口美國牛肉", 13: "台灣不應調漲基本薪資" }
    
    mediaMap = getMediaMap(lnInTopic)
    for topicId, lns in sorted(lnInTopic.items(), key=lambda x:x[0]):
        print('Topic:', topicId, statName[topicId])
        cntList = [[0 for i in range(0, len(mediaMap))] for j in range(0, len(i2Label))]
        for ln in lns:
            media = getMedia(ln['news_id'])
            label = ln['label']
            cntList[label2i[label]][mediaMap[media]] += 1
        
        for media, i in sorted(mediaMap.items(), key=lambda x:x[1]):
            print(',', mediaName[media], end='')
        print('')
        for i, cnt in enumerate(cntList):
            print(i2Label[i], end=',')
            for c in cnt:
                print(c, end=',')
            print('')
        print('\n')

def getMediaMap(lnList):
    mediaMap = dict()
    for topicId, lnList in lnInTopic.items():
        for ln in lnList:
            media = getMedia(ln['news_id'])
            if media not in mediaMap:
                mediaMap[media] = len(mediaMap)
    return mediaMap

def getMedia(newsId):
    return newsId[0:newsId.rfind('_')]



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'labelNewsJsonFile config', file=sys.stderr)
        exit(-1)

    lnFile = sys.argv[1]
    modelConfigFile = sys.argv[2]

    # load model config
    with open(modelConfigFile, 'r') as f:
        config = json.load(f)

    with open(lnFile, 'r') as f:
        lnList = json.load(f)
    lnInTopic = divideLabelNewsByTopic(lnList)
    topicSet = set([ln['statement_id'] for ln in lnList])
    topicMap = [ lnList[i]['statement_id'] for i in range(0, len(lnList)) ]
    newsIdList = { t:[ln['news_id'] for ln in lnInTopic[t]] for t in topicSet }
    newsIdList['All'] = [ln['news_id'] for ln in lnList] 
    
    taskName = config['taskName']
    setting = config['setting']
    paramIter = ParameterGrid(config['params'])

    mediaVolc = getMediaVolc(lnInTopic)
    for t, lns in sorted(lnInTopic.items(), key=lambda x:x[0]):
        for p in paramIter:
            (X, y) = genXY(lns, mediaVolc)
            expLog = RunExp.runTask(X, y, { "main": mediaVolc } , newsIdList[t], 'SelfTrainTest', p, topicId=t, **setting)
            with open('%s_SelfTrainTest_topic%d.pickle' % (taskName, t), 'w+b') as f:
                pickle.dump(expLog, f)


