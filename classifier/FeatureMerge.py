
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

'''
The following functions are for generating feature-to-feature graph by calculating pairwise cosine similarity
'''
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

'''
The following functions are for generating auxiliary vectors
'''
# firstly divide features into groups, and calculate each feature vectors by word vector. 
# return: 
def getFeatureVectorsByGroup(volc, wordVector, version):
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
            # if feature vector cannot be calculated, it will return None
            word = volc.getWord(i)
            vector = calcFeatureVector(word, wordVector, version)
            if vector is not None:
                vectors.append(vector)
                newVolc.addWord(volc.getWord(i))
                mapping.append(i)
            else:
                vectors.append(None)
                newVolc.addWord(None)
                mapping.append(i)
        groupVectors[fType] = np.array(vectors)
        groupMapping[fType] = mapping
        groupVolc[fType] = newVolc
    return groupVectors, groupVolc, groupMapping

def divideFeatureIntoGroup(volc):
    groups = dict()
    for i in range(0, len(volc)):
        v = volc.getWord(i)
        fType = (v)
        if fType not in groups:
            groups[fType] = list()
        groups[fType].append(i)
    return groups

# get all feature vectors
def getFeatureVectors(volc, wordVector, version):
    vectors = list() # feature vectors
    mapping = list() # mapping[i] is the original index if vectors[i]
    newVolc = Volc() # newVolc
    for i in range(0, len(volc)):
        word = volc.getWord(i)
        vector = AuxiliaryVectorHelper.calcFeatureVector(word, wordVector, version)
        if vector is not None:
            vectors.append(vector)
            newVolc.addWord(volc.getWord(i))
            mapping.append(i)
    return np.array(vectors), newVolc, mapping

''' Depricated on 2016/2/1
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
'''

def __getSign(string):
    if string.find('sign') == -1:
        return None
    else:
        return int(string[string.find('sign')+4:])



''' 
         version 1    version 2
Word      Word         Word
BiWord    BiWord       BiWord
TriWord   TriWord      TriWord
T & 0     H/T_N        H/T_0
T & 1     H/T_P        H/T_P
T &-1     H/T_P        H/T_N
H & 0     H/T_N        H/T_0
H & 1     H/T_P        H/T_P
H &-1     H/T_P        H/T_N
HO& 1     HO/OT        HO/OT_P
HO&-1     HO/OT        HO/OT_N
OT& 1     HO/OT        HO/OT_P
OT&-1     HO/OT        HO/OT_P
HOT         X            X
          P=polarity   P=Positive
          N=neutral    N=Negative
                       0=Neutral
In version 1, positive and negative are put in same group, feature vector is multiplied by -1
In version 2, positive and negative are put in different groups, feature vector is the same
Date: 2016/2/1
'''
class AuxiliaryVectorHelper:
    def calcFeatureVector(word, wordVector, version):
        if word is None:
            return None

        fType = self.getFeatureType(word) # feature type
        vector = None

        # for word-based features, we don't need to consider version
        if fType in ['Word', 'BiWord', 'TriWord']: 
            if fType == 'Word':
                vector = np.array(wordVector[v]) if v in wordVector else vector
            else:
                for vi in v:
                    if vi not in wordVector: 
                        continue
                    if vector is None: 
                        vector = np.array(wordVector[vi])
                    else: 
                        vector += wordVector[vi]

        # for dependency features, we have to consider version
        else:             
            if version == 1:
                if fType in ['H_P', 'T_P', 'H/T_P']: 
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
            elif version == 2:
                if fType in ['H/T_P', 'H/T_N', 'H/T_0']:
                    vector = wordVector[v[1]] if v[1] in wordVector else vector
                elif fType in ['HO/OT_P', 'HO/OT_N']:
                    for i in [1, 3]:
                        if v[i] not in wordVector: continue
                        if vector is None: 
                            vector = np.array(wordVector[v[i]])
                        else:
                            vector += wordVector[v[i]]

        return vector

    # According to feature form, get the feature type. There are 2 versions. 
    def getFeatureType(self, word, version):
        if version == 1:
            if type(v) == str:
                return 'Word'
            elif type(v) == tuple:
                if v[0] in ['T', 'H'] and len(v) >= 3 and __getSign(v[2]) is not None:
                    sign = __getSign(v[2])
                    if sign == 0: return 'H/T_N'
                    else: return 'H/T_P'
                elif v[0] in ['OT', 'HO']: #not support HT HOT
                    return 'HO/OT'
                elif len(v) == 2:
                    return 'BiWord'
                elif len(v) == 3:
                    return 'TriWord'
            else:
                return None
        elif version == 2:
            if type(v) == str:
                return 'Word'
            elif type(v) == tuple:
                if v[0] in ['T', 'H'] and len(v) >= 3 and __getSign(v[2]) is not None:
                    sign = __getSign(v[2])
                    if sign == 0: 
                        return v[0] + '_0'
                    elif sign == 1: 
                        return v[0] + '_P'
                    else:
                        return v[0] + '_N'
                elif v[0] in ['OT', 'HO']: #not support HT HOT
                    sign = __getSign(v[2])
                    if sign == 1:
                        return 'HO/OT_P'
                    elif sign == -1:
                        return 'HO/OT_N'
                elif len(v) == 2:
                    return 'BiWord'
                elif len(v) == 3:
                    return 'TriWord'
            else:
                return None



''' Depricated on 2016/2/1
# T_N: target & sign = 0
# H_N: holder & sign = 0
# T_P: target & sign = +1/-1
# H_P: target & sign = +1/-1
# H/T_N: holder or target & sign = 0
# H/T_P: holder or target & sign = +1/-1
typeMappingV1 = { 'T_N': 'H/T_N', 'H_N':'H/T_N', 'T_P':'H/T_P', 'H_P':'H/T_P',  'OT': 'HO/OT', 'HO': 'HO/OT' }
#typeMappingV2 = { 'T_N': 'T_N', 'H_N':'H_N', 'T_P':'T_P', 'H_P':'H_P',  'OT': 'OT', 'HO': 'HO' }
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
'''

'''
The following functions are for clustering features (community detection or KMeans)
'''
# given coefficient(weight) of classifier and feature adjacency list
# generate projection model for mergin features
def featureClustering(coef, volc, adjSet):
    posSet, negSet = dividePosNegSet(coef)
    posClusters = clusterFeatures(posSet, adjSet)    
    negClusters = clusterFeatures(negSet, adjSet)
    model = genFeatureMergingModel(posClusters, negClusters, volc)
    return model

# nClusters can be:
#   integer: #centroid for each kind of features 
#   float: percentage of centroid for each kind of features
#   dict: feature type -> int or float
def convertNCluster(fTypes, nClusters):
    if type(nClusters) == int or type(nClusters) == float:
        nC = { fType:nClusters for fType in fTypes}
    elif type(nClusters) == dict:
        assert len(set(fTypes) - set(nClusters.keys())) == 0
        for fType, nc in nClusters.items():
            assert type(nc) == int or type(nc) == float
        nC = dict(nClusters)
    else:
        return None
    return nC

# Divide features into groups -> KMeans clustering 
# groupVectors: featureType -> featureVectors (matrix or np array?)
# groupMapping?
# nClusters can be:
#   integer: #centroid for each kind of features 
#   float: percentage of centroid for each kind of features
#   dict: feature type -> int or float
def featureClustering_KMeans_byGroup(groupVectors, groupMapping, nClusters, max_iter=300):
    nC = convertNCluster(set(groupVectors.keys()), nClusters)
    finalClusters = list()
    oriDim = 0
    for fType, vectors in groupVectors.items():
        print('feature type:', fType, ' #features:', vectors.shape[0], file=sys.stderr)
        groupSet = set([i for i in range(0, len(vectors))])
        clusters = clusterFeatures_KMeans(groupSet, vectors, groupMapping[fType], nC[fType], max_iter)
        finalClusters.extend(clusters)
        oriDim += len(vectors)

    checkClusters(finalClusters)
    model = FeatureMergingModel(clusters, oriDim)
    return model


def featureClustering_KMeans_byGroup_TwoSide(coef, groupVectors, groupMapping, nClusters, max_iter=100):
    posSet, negSet = dividePosNegSet(coef)
    
    nC = convertNCluster(set(groupVectors.keys()), nClusters)
    finalClusters = list()
    oriDim = 0
    for fType, vectors in groupVectors.items():
        oriDim += vectors.shape[0]

    for fType, vectors in groupVectors.items():
        print('feature type:', fType, ' #features:', vectors.shape[0], file=sys.stderr)
        mapping = groupMapping[fType]
        invMap = { old:new for new, old in enumerate(mapping) }
        
        gPosSet = set(mapping) & posSet
        gPosSet = set([invMap[i] for i in gPosSet])
        gNegSet = set(mapping) & negSet
        gNegSet = set([invMap[i] for i in gNegSet])
        pNum = len(gPosSet)
        nNum = len(gNegSet)
        
        nC1 = nC[fType] * float(pNum) / (pNum + nNum)
        nC2 = nC[fType] * float(nNum) / (pNum + nNum)
        print(nC1, nC2, file=sys.stderr)
        clusters = clusterFeatures_KMeans(gPosSet, vectors, groupMapping[fType], nC1, max_iter)
        finalClusters.extend(clusters)

        clusters = clusterFeatures_KMeans(gNegSet, vectors, groupMapping[fType], nC2, max_iter)
        finalClusters.extend(clusters)
    
    checkClusters(finalClusters)
    model = FeatureMergingModel(finalClusters, oriDim)
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

# clustering features by KMeans 
# nClusters: if nClusters > 1 then nClusters is the number of cluster centers
#   otherwise, nClusters is the pecentage of original feature number 
#   when the number is too small, at least set to 1 
# Why there is none vector? Because some of auxiliary vectors cannot be generated, words are not in the result from word2vec tool
def clusterFeatures_KMeans(groupSet, vectors, oriMapping, nClusters, max_iter=100):
    nClusters = nClusters if nClusters > 1 else int(nClusters * len(groupSet))
    nClusters = nClusters if nClusters > 1 else 1

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
    nC = nClusters - len(remainSet)
    #print(nC, nClusters, len(remainSet))
    nC = nC if nC > 1 else 2
    est = KMeans(n_clusters=nC, max_iter=max_iter)
    label = est.fit_predict(X)
    clusters = [list() for i in range(0, nC)]
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

    # read word vectors
    volc, vectors = readWordVector(wordVectorFile)
    wordVector = toDictType(volc, vectors)

    # read pickle file (volc)
    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)
    volc = p['mainVolc']
    print('# feature:', len(volc), file=sys.stderr)

    groupVectors, groupVolc, groupMapping = getFeatureVectorsByGroup(volc, wordVector, version)
    adjListList = calcSimAndFilterAdjListByGroup(groupVectors, groupMapping, thresholdList, len(volc))
    
    for i, adjList in enumerate(adjListList):
        filename = outPrefix + '_T%g.adjList' % (thresholdList[i])
        saveAdjList(filename, adjList)
    #if outAdjWordFile is not None:
    #    saveAdjWordFile(outAdjWordFile, adjList, volc)

