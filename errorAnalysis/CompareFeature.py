
import sys, pickle

def checkClusters(clusters):
    merged = set()
    for i in range(0, len(clusters)):
        s = set(clusters[i])
        intersect = merged & s
        if len(intersect) > 0:
            print(intersect, file=sys.stderr)
            print('Cluster Error', file=sys.stderr)
            return False
        merged.update(s)
    return True

def convert2Clusters(model):
    X = model.proj
    (rowNum, colNum) = X.shape
    colIndex, rowPtr, data = X.indices, X.indptr, X.data
    clusters = [list() for i in range(colNum)]
    for ri in range(0, rowNum):
        for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
            clusters[ci].append(ri)
    assert checkClusters(clusters)
    return clusters

# given coefficient(weight) of classifier, divide feature into two group, 
# (positvie and negative), assume binary classification
# posList: list of index where weights >= 0
def getPosNegMapAndRanking(coef):
    mapping = list()
    posList = list()
    negList = list()
    for i, w in enumerate(coef[0]):
        if w >= 0.0:
            mapping.append(1)
            posList.append((i, w))
        else:
            mapping.append(-1)
            negList.append((i, -w))
    posList.sort(key=lambda x:x[1], reverse=True)
    negList.sort(key=lambda x:x[1], reverse=True)
    ranking = [0 for i in range(0, len(coef[0]))]
    for r, (i, w) in enumerate(posList):
        ranking[i] = r
    for r, (i, w) in enumerate(negList):
        ranking[i] = r
    return mapping, ranking

def getFalseClusteredFeatures(mapping, ranking, clusters):
    falsePairs = list()
    for cluster in clusters:
        for i in range(0, len(cluster)):
            for j in range(i+1, len(cluster)):
                k1 = cluster[i]
                k2 = cluster[j]
                if mapping[k1] != mapping[k2]:
                    falsePairs.append((k1, k2, ranking[k1]+ranking[k2]))

    falsePairs.sort(key=lambda x:x[2])
    return falsePairs

def getMinorProportion(mapping, ranking, clusters):
    avg = 0.0
    cNum = 0
    for cluster in clusters:
        if len(cluster) != 0:
            cnt = { 1: 0, -1: 0 }
            for i in range(0, len(cluster)):
                cnt[mapping[i]] += 1
            avg += min(cnt.values()) / sum(cnt.values())
            cNum += 1
    avg = avg / cNum
    return avg

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:', sys.argv[0], 'Ori_LogPickle DirectFC_LogPickle [dataPickle]', file=sys.stderr) 
        exit(-1)

    with open(sys.argv[1], 'r+b') as f:
        ori_P = pickle.load(f)
    with open(sys.argv[2], 'r+b') as f:
        DFC_P = pickle.load(f)

    dataP = None
    if len(sys.argv) == 4:
        with open(sys.argv[3], 'r+b') as f:
            dataP = pickle.load(f)

    clusters = convert2Clusters(DFC_P[0]['model'])
    coef = ori_P[0]['clf'].coef_
    mapping, ranking = getPosNegMapAndRanking(coef)
    p = getMinorProportion(mapping, ranking, clusters)

    if dataP is not None:
        falsePairs = getFalseClusteredFeatures(mapping, ranking, clusters)
        volc = dataP['mainVolc']
        for pair in falsePairs:
            print(volc.getWord(pair[0]), volc.getWord(pair[1]), pair[2])

    print(p)

