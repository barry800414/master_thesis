
import sys, math, pickle, copy
from WordMerge import *
from sklearn.cluster import KMeans

class FeatureMergingModel():
    def __init__(self, clusters, oriDim):
        self.clusters = copy.deepcopy(clusters)
        # generating projection matrix
        rows, cols, data = list(), list(), list()
        for i, cluster in enumerate(clusters):
            for j in cluster:
                assert j < oriDim
                rows.append(j)
                cols.append(i)
                data.append(1)
        self.proj = csr_matrix((data, (rows, cols)), shape=(oriDim, len(clusters)), dtype=np.int8)
        
    def transform(self, X):
        return X * self.proj

    def toDim(self):
        return self.proj.shape[1]


# given feature vectors, calculating pair-wise cosine similarity
# and filter out the edges whose sim < threshold
# adjList: adjacency list
# adjListList: list of adjacency list
def calcSimAndFilterAdjListByGroup(groupVectors, groupMapping, thresholdList, nFeature):
    finalAdjListList = None
    for fType, vectors in groupVectors.items():
        print('feature type:', fType, ' #features:', vectors.shape[0], file=sys.stderr)
        print('calculating pairwise cosine similarity ...', file=sys.stderr)
        dist = pdist(vectors, 'cosine')

        # get list of selected edges by given method
        print('Filtering edges ...', end='', file=sys.stderr)
        adjListList, edgeNumList = getAdjListList(dist, vectors.shape[0], thresholdList, groupMapping[fType], nFeature)
        for i, threshold in enumerate(thresholdList):
            edgeNum = edgeNumList[i]
            print('Threshold:', threshold, ' #edges:', edgeNum, ' avgEdgeNum:', edgeNum/len(volc), file=sys.stderr)
        finalAdjListList = mergeAdjListList(finalAdjListList, adjListList)
    return finalAdjListList

def mergeAdjListList(target, source):
    if target is None:
        return copy.deepcopy(source)
    else:
        assert len(target) == len(source)
        for i in range(0, len(target)):
            for j in range(0, len(target[i])):
                if len(source[i][j]) != 0:
                    assert len(target[i][j]) == 0
                target[i][j].extend(source[i][j])
        return target

# given feature vectors, calculating pair-wise cosine similarity
# and filter out the edges whose sim < threshold
# adjList: adjacency list
# adjListList: list of adjacency list
def calcSimAndFilterAdjList(vectors, mapping, thresholdList, nFeature):
    print('calculating pairwise cosine similarity ...', file=sys.stderr)
    dist = pdist(vectors, 'cosine')

    # get list of selected edges by given method
    print('Filtering edges ...', file=sys.stderr)
    adjListList, edgeNumList = getAdjListList(dist, vectors.shape[0], thresholdList, mapping, nFeature)
    
    for i, threshold in enumerate(thresholdList):
        edgeNum = edgeNumList[i]
        print('Threshold:', threshold, ' #edges:', edgeNum, ' avgEdgeNum:', edgeNum/nFeature, file=sys.stderr)
    return adjListList

def getAdjListList(dist, nFeatureOfVector, thresholdList, mapping, nFeature):
    tNum = len(thresholdList)
    adjListList = [[list() for i in range(0, nFeature)] for j in range(0, tNum)]
    index = 0
    cntList = [0 for i in range(0, tNum)]
    for i in range(0, nFeatureOfVector):
        for j in range(i+1, nFeatureOfVector):
            sim = 1. - dist[index]
            for k, threshold in enumerate(thresholdList):
                if sim > threshold:
                    adjListList[k][mapping[i]].append(mapping[j])
                    cntList[k] += 1
            index += 1
    return adjListList, cntList


def getAdjList(dist, nFeatureOfVector, thresholdList, mapping, nFeature):
    adjList = [list() for i in range(0, nFeature)]
    index = 0
    cnt = 0
    for i in range(0, nFeatureOfVector):
        for j in range(i+1, nFeatureOfVector):
            sim = 1. - dist[index]
            if sim > threshold:
                adjList[mapping[i]].append(mapping[j])
                cnt += 1
            index += 1
    return adjList, cnt

# get all feature vectors
def getFeatureVectorsByGroup(volc, wordVector):
    # first divide feature into several groups
    groups = divideFeatureIntoGroup(volc)
    print(groups.keys(), file=sys.stderr)
    groupVectors = dict()
    groupMapping = dict()
    groupVolc = dict()
    for fType, idList in groups.items():
        vectors = list() # feature vectors
        mapping = list() # mapping[i] is the original index if vectors[i]
        newVolc = Volc() # newVolc
        for i in idList:
            vector = calcFeatureVector(i, volc, wordVector)
            if vector is not None:
                vectors.append(vector)
                newVolc.addWord(volc.getWord(i))
                mapping.append(i)
        groupVectors[fType] = np.array(vectors)
        groupMapping[fType] = mapping
        groupVolc[fType] = newVolc
    return groupVectors, groupVolc, groupMapping

def divideFeatureIntoGroup(volc):
    groups = dict()
    for i in range(0, len(volc)):
        v = volc.getWord(i)
        fType = checkType(v)
        if fType not in groups:
            groups[fType] = list()
        groups[fType].append(i)
    return groups

# get all feature vectors
def getFeatureVectors(volc, wordVector):
    vectors = list() # feature vectors
    mapping = list() # mapping[i] is the original index if vectors[i]
    newVolc = Volc() # newVolc
    for i in range(0, len(volc)):
        vector = calcFeatureVector(i, volc, wordVector)
        if vector is not None:
            vectors.append(vector)
            newVolc.addWord(volc.getWord(i))
            mapping.append(i)
    return np.array(vectors), newVolc, mapping

# calculate one feature vector
def calcFeatureVector(index, volc, wordVector):
    v = volc.getWord(index) # volcabulary 
    fType = checkType(v) # feature type
    vector = None
    if fType == 'Word':
        vector = np.array(wordVector[v]) if v in wordVector else vector
    elif fType in ['H_P', 'T_P', 'H/T_P']: 
        assert len(v) == 3
        sign = __getSign(v[2])
        vector = sign * wordVector[v[1]] if v[1] in wordVector else vector           
    elif fType in ['H_N', 'T_N', 'H/T_N']:
        vector = wordVector[v[1]] if v[1] in wordVector else vector
    elif fType in ['OT', 'HO', 'HO/OT']:
        sign = __getSign(v[2])
        for i in [1, 3]:
            if v[i] not in wordVector: continue
            if vector is None: 
                vector = np.array(wordVector[v[i]])
            else:
                vector += wordVector[v[i]]
        if vector is not None: vector = vector * sign
    elif fType in ['HT']: 
        pass
    elif fType in ['HOT']:
        pass
    elif fType in ['BiWord', 'TriWord']:
        for vi in v:
            if vi not in wordVector: 
                #print(vi)
                continue
            if vector is None: 
                vector = np.array(wordVector[vi])
            else: 
                vector += wordVector[vi]
    return vector


def __getSign(string):
    if string.find('sign') == -1:
        return None
    else:
        return int(string[string.find('sign')+4:])

# T_N: target & sign = 0
# H_N: holder & sign = 0
# T_P: target & sign = +1/-1
# H_P: target & sign = +1/-1
# H/T_N: holder or target & sign = 0
# H/T_P: holder or target & sign = +1/-1
typeMappingV1 = { 'T_N': 'H/T_N', 'H_N':'H/T_N', 'T_P':'H/T_P', 'H_P':'H/T_P',  'OT': 'HO/OT', 'HO': 'HO/OT' }
typeMappingV2 = { 'T_N': 'T_N', 'H_N':'H_N', 'T_P':'T_P', 'H_P':'H_P',  'OT': 'OT', 'HO': 'HO' }
typeMapping = typeMappingV1

def checkType(v):
    if type(v) == str:
        return 'Word'
    elif type(v) == tuple:
        if v[0] in ['T', 'H'] and len(v) >= 3 and __getSign(v[2]) is not None:
            sign = __getSign(v[2])
            if sign == 0:
                return typeMapping[v[0] + '_N']
            else:
                return typeMapping[v[0] + '_P']
        elif v[0] in ['OT', 'HO', 'HT', 'HOT']: #not support HT HOT
            return typeMapping[v[0]]
        elif len(v) == 2:
            return 'BiWord'
        elif len(v) == 3:
            return 'TriWord'
    else:
        return None

def setType(version):
    global typeMapping
    if version == 1:
        typeMapping = typeMappingV1
    elif version == 2:
        typeMapping = typeMappingV2

# given coefficient(weight) of classifier and feature adjacency list
# generate projection model for mergin features
def featureClustering(coef, volc, adjSet):
    posSet, negSet = dividePosNegSet(coef)
    posClusters = clusterFeatures(posSet, adjSet)    
    negClusters = clusterFeatures(negSet, adjSet)
    model = genFeatureMergingModel(posClusters, negClusters, volc)
    return model


def featureClustering_KMeans_byGroup(groupVectors, groupMapping, nClusters, max_iter=300):
    finalClusters = list()
    oriDim = 0
    for fType, vectors in groupVectors.items():
        print('feature type:', fType, ' #features:', vectors.shape[0], file=sys.stderr)
        groupSet = set([i for i in range(0, len(vectors))])
        clusters = clusterFeatures_KMeans(groupSet, vectors, groupMapping[fType], nClusters, max_iter)
        finalClusters.extend(clusters)
        oriDim += len(vectors)

    checkClusters(finalClusters)
    model = FeatureMergingModel(clusters, oriDim)
    return model


def featureClustering_KMeans_byGroup_TwoSide(coef, groupVectors, groupMapping, nClusters, max_iter=300):
    posSet, negSet = dividePosNegSet(coef)

    finalClusters = list()
    oriDim = 0
    for fType, vectors in groupVectors.items():
        print('feature type:', fType, ' #features:', vectors.shape[0], file=sys.stderr)
        mapping = groupMapping[fType]
        invMap = { old:new for new, old in enumerate(mapping) }
        
        groupSet = set(mapping) & posSet
        groupSet = set([invMap[i] for i in groupSet])
        clusters = clusterFeatures_KMeans(groupSet, vectors, groupMapping[fType], nClusters, max_iter)
        finalClusters.extend(clusters)

        groupSet = set(mapping) & negSet
        groupSet = set([invMap[i] for i in groupSet])
        clusters = clusterFeatures_KMeans(groupSet, vectors, groupMapping[fType], nClusters, max_iter)
        finalClusters.extend(clusters)
    
        oriDim += len(vectors)

    checkClusters(finalClusters)
    model = FeatureMergingModel(clusters, oriDim)
    return model

    
# given coefficient(weight) if classifier, divide feature into two group, 
# (positvie and negative), assume binary classification
# posList: list of index where weights >= 0
def dividePosNegSet(coef):
    posSet = set()
    negSet = set()
    for i, w in enumerate(coef[0]):
        if w >= 0.0:
            posSet.add(i)
        else:
            negSet.add(i)
    return posSet, negSet

# given feature vectors, do feature clustering
# group set: the set of index in that group
# adjSet[i]: the edge set of i-th node
def clusterFeatures(groupSet, adjSet):
    g = nx.Graph()
    remainSet = set(groupSet)
    for i in groupSet:
        toSet = adjSet[i] & groupSet
        for j in toSet:
            g.add_edge(i, j)
        if len(toSet) != 0:
            remainSet.difference_update(toSet)
            remainSet.discard(i)
    
    if g.number_of_edges() > 0:
        partition = community.best_partition(g)
        clusters = convert2NodeList(partition)
    else:
        clusters = list()

    for i in remainSet:
        clusters.append([i])
    return clusters

def clusterFeatures_KMeans(groupSet, vectors, oriMapping, nClusters, max_iter=300):
    nClusters = nClusters if nClusters > 1 else int(nClusters * len(groupSet))

    X = list()
    remainSet = set()
    mapping = list()
    for i in groupSet:
        if vectors[i] is not None:
            X.append(vectors[i])
            mapping.append(oriMapping[i])
        else:
            remainSet.add(oriMapping[i])
    X = np.array(X)

    print('Running KMeans ... n_clusters:', nClusters, ' max_iter:', max_iter, file=sys.stderr)
    est = KMeans(n_clusters=nClusters, max_iter=max_iter)
    label = est.fit_predict(X)
    clusters = [list() for i in range(0, nClusters)]
    for i, l in enumerate(label):
        clusters[l].append(mapping[i])

    for i in remainSet:
        clusters.append([i])

    return clusters


def genFeatureMergingModel(posClusters, negClusters, volc):
    clusters = copy.deepcopy(posClusters)
    clusters.extend(negClusters)
    checkClusters(clusters)
    model = FeatureMergingModel(clusters, len(volc))
    return model


def saveAdjList(filename, adjList):
    with open(filename, 'w') as f:
        for fromId, toNodes in enumerate(adjList):
            for toId in toNodes:
                if toId < fromId: continue
                print(toId, end=' ', file=f)
            print('', file=f)

def readAdjList(filename, offset=0):
    adjSet = list()
    with open(filename, 'r') as f:
        for fromId, line in enumerate(f):
            toIds = line.strip().split()
            toSet = set()
            for toId in toIds:
                toSet.add(int(toId) + offset)
            adjSet.append(toSet)
    return adjSet

def saveAdjWordFile(filename, adjList, volc):
    with open(filename, 'w') as f:
        for fromId, toNodes in enumerate(adjList):
            print(volc.getWord(fromId), end=':', file=f)
            for toId in toNodes:
                print(volc.getWord(toId), end=' ', file=f)
            print('\n', file=f)


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage:', sys.argv[0], 'WordVector(.vector) pickleFile outPrefix version threshold1 threshold2 ...', file=sys.stderr)
        exit(-1)

    wordVectorFile = sys.argv[1]
    pickleFile = sys.argv[2]
    outPrefix = sys.argv[3]
    version = int(sys.argv[4])
    thresholdList = list()
    for i in range(5, len(sys.argv)):
        thresholdList.append(float(sys.argv[i]))

    # set version 
    setType(version)
    print('typeMapping:', typeMapping, file=sys.stderr)

    # read word vectors
    volc, vectors = readWordVector(wordVectorFile)
    wordVector = toDictType(volc, vectors)

    # read pickle file (volc)
    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)
    volc = p['mainVolc']
    print('# feature:', len(volc), file=sys.stderr)

    groupVectors, groupVolc, groupMapping = getFeatureVectorsByGroup(volc, wordVector)
    adjListList = calcSimAndFilterAdjListByGroup(groupVectors, groupMapping, thresholdList, len(volc))
    
    for i, adjList in enumerate(adjListList):
        filename = outPrefix + '_T%g.adjList' % (thresholdList[i])
        saveAdjList(filename, adjList)
    #if outAdjWordFile is not None:
    #    saveAdjWordFile(outAdjWordFile, adjList, volc)

