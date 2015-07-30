
import sys, math, pickle
from WordMerge import *

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


# given feature vectors, calculating pair-wise cosine similarity
# and filter out the edges whose sim < threshold
# adjList: adjaceny list
def calcSimAndFilterAdjList(vectors, volc, mapping, threshold):
    print('calculating pairwise cosine similarity ...', file=sys.stderr)
    dist = pdist(vectors, 'cosine')

    # get list of selected edges by given method
    print('Filtering edges ...', end='', file=sys.stderr)
    adjList, edgeNum = getAdjList(dist, len(volc), threshold, mapping) # list of (sim, (x1, x2))
    
    print('#edges:', edgeNum, ' avgEdgeNum:', edgeNum/len(volc), file=sys.stderr)

    return adjList

def getAdjList(dist, nFeature, threshold, mapping):
    adjList = [list() for i in range(0, nFeature)]
    index = 0
    cnt = 0
    for i in range(0, nFeature):
        for j in range(i+1, nFeature):
            sim = 1. - dist[index]
            if sim > threshold:
                adjList[i].append(mapping[j])
                cnt += 1
            index += 1
    return adjList, cnt

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
    elif fType in ['T', 'H']: #TODO
        assert len(v) == 3
        sign = __getSign(v[2])
        vector = sign * wordVector[v[1]] if v[1] in wordVector else vector           
        #print(sign)
    elif fType in ['OT', 'HO']:
        sign = __getSign(v[2])
        #print(sign)
        for i in [1, 3]:
            if v[i] not in wordVector: continue
            if vector is None: 
                vector = np.array(wordVector[v[i]])
            else:
                vector += wordVector[v[i]]
        if vector is not None: vector = vector * sign
    elif fType in ['HT']: #TODO
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
    return int(string[string.find('sign')+4:])

def checkType(v):
    if type(v) == str:
        return 'Word'
    elif type(v) == tuple:
        if v[0] in ['T', 'OT', 'HO', 'H', 'HT', 'HOT']:
            return v[0]
        elif len(v) == 2:
            return 'BiWord'
        elif len(v) == 3:
            return 'TriWord'
    else:
        return None

def saveAdjList(filename, adjList):
    with open(filename, 'w') as f:
        for fromId, toNodes in enumerate(adjList):
            for toId in toNodes:
                if toId < fromId: continue
                print(toId, end=' ', file=f)
            print('', file=f)

def saveAdjList(filename, adjList):
    with open(filename, 'w') as f:
        for fromId, toNodes in enumerate(adjList):
            for toId in toNodes:
                if toId < fromId: continue
                print(toId, end=' ', file=f)
            print('', file=f)

def saveAdjWordFile(filename, adjList, volc):
    with open(filename, 'w') as f:
        for fromId, toNodes in enumerate(adjList):
            print(volc.getWord(fromId), end=':', file=f)
            for toId in toNodes:
                print(volc.getWord(toId), end=' ', file=f)
            print('\n', file=f)


# given coefficient(weight) of classifier and feature adjacency list
# generate projection model for mergin features
def featureClustering(coef, volc, adjSet, threshold):
    posSet, negSet = dividePosNegSet(coef)
    
    # posClusters
    # negClusteres

    model = ML.genFeatureMergingModel(posClusters, negClusters, volc)
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
    remainSet = set()
    for i in groupSet:
        toSet = adjSet[i] & groupSet
        for j in toSet:
            g.add_edge(i, j)
        if len(toSet) != 0:
            remainSet.difference_update(toSet)
            remainSet.remove(i)

    partition = community.best_partition(g)
    clusters = convert2NodeList(partition)

    for i in remainSet:
        clusters.append([i])
    return clusters

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage:', sys.argv[0], 'WordVector(.vector) pickleFile threshold OutAdjFile [OutAdjWordFile]', file=sys.stderr)
        exit(-1)

    wordVectorFile = sys.argv[1]
    pickleFile = sys.argv[2]
    threshold = float(sys.argv[3])
    outAdjFile = sys.argv[4]
    outAdjWordFile = sys.argv[5] if len(sys.argv) == 6 else None

    # read word vectors
    volc, vectors = readWordVector(wordVectorFile)
    wordVector = toDictType(volc, vectors)

    # read pickle file (volc)
    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)
    volc = p['mainVolc']

    vectors, newVolc, mapping = getFeatureVectors(volc, wordVector)
    adjList = calcSimAndFilterAdjList(vectors, newVolc, mapping, threshold)
    
    saveAdjList(outAdjFile, adjList)
    if outAdjWordFile is not None:
        saveAdjWordFile(outAdjWordFile, adjList, volc)

