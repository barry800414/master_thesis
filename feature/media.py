
import sys, json, pickle

from scipy.sparse import csr_matrix
import numpy as np

from Volc import Volc
from misc import *

def genX(lnList, mediaVolc):
    mIdList = list()
    volc = Volc()
    for ln in lnList:
        media = getMedia(ln['news_id'])
        if media not in volc:
            volc.addWord(media)
        mIdList.append(volc[media])

    rows, cols, data = list(), list(), list()
    for i, mId in enumerate(mIdList):
        rows.append(i)
        cols.append(mId)
        data.append(1)
    X = csr_matrix((data, (rows, cols)), shape=(len(lnList), len(volc)))
    return X, volc

def getMediaVolc(lnList):
    mediaVolc = Volc()
    for topicId, lnList in lnListInTopic.items():
        for ln in lnList:
            media = getMedia(ln['news_id'])
            mediaVolc.addWord(media)
    return mediaVolc

def printMediaStat(lnListInTopic):
    mediaName = { '02_coolloud':"苦勞", '07_setn':"三立", '05_ltn':'自由', '03_stormmedia':"風傳", '06_udn':'聯合', '01_pnn':'公視', '04_thenewslens':'關鍵' }
    statName = { 2: "台灣不應停建核四", 3: "應簽訂服務貿易協議", 4: "應簽訂自由經濟示範區條例", 5: "台灣應進口美國牛肉", 13: "台灣不應調漲基本薪資" }
    
    mediaMap = getMediaMap(lnListInTopic)
    for topicId, lns in sorted(lnListInTopic.items(), key=lambda x:x[0]):
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

def getMediaMap(lnListInTopic):
    mediaMap = dict()
    for topicId, lnList in lnListInTopic.items():
        for ln in lnList:
            media = getMedia(ln['news_id'])
            if media not in mediaMap:
                mediaMap[media] = len(mediaMap)
    return mediaMap

def getMedia(newsId):
    return newsId[0:newsId.rfind('_')]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], 'labelNewsJsonFile', file=sys.stderr)
        exit(-1)

    lnFile = sys.argv[1]

    with open(lnFile, 'r') as f:
        lnList = json.load(f)
    lnListInTopic = divideLabelNewsByTopic(lnList)
    mediaVolc = getMediaVolc(lnListInTopic)

    for t, lnList in sorted(lnListInTopic.items(), key=lambda x:x[0]):
        (labelIndex, unLabelIndex) = getLabelIndex(lnList)
        labelLnList = [lnList[i] for i in labelIndex]
        
        X, volc = genX(lnList, mediaVolc)
        ally = np.array(getLabels(lnList))
        y = ally[labelIndex]
        pObj = { 'X':X, 'unX': None, 'y':y, 'mainVolc': volc, 'config': 'media' }
        with open('t%d_media.pickle' % (t),'w+b') as f:
            pickle.dump(pObj, f)

