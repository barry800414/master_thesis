
import sys, math

import numpy as np
import networkx as nx
import community

import WordTag
from scipy.spatial.distance import pdist
from misc import *
from Volc import Volc

# read word vector(text file) from word2vec tool
def readWordVector(filename, allowedWord=None):
    volc = Volc()
    with open(filename, 'r') as f:
        line = f.readline()
        entry = line.strip().split(' ')
        volcNum = int(entry[0])
        dim = int(entry[1])
        vectors = list()
        i = 0
        while True:
            try:
                line = f.readline()
                if not line:
                    break
            except Exception as e:
                print('\nAt line %d' % i, e)
                line = f.readline()
                volcNum = volcNum - 1
                i = i + 1
                continue
            
            entry = line.strip().split(' ')
            if len(entry) != dim + 1:
                print(line)
                print(len(entry))
            assert len(entry) == dim + 1
            w = entry[0].strip()
            if allowedWord is not None and w not in allowedWord:
                continue
            volc.addWord(w)
            vector = list()
            for j in range(1, len(entry)):
                vector.append(float(entry[j]))
            vectors.append(vector)
            i = i + 1
            if (i+1) % 10000 == 0:
                print('%cProgress: (%d/%d)' % (13, i+1, volcNum), end='', file=sys.stderr)
        print('', file=sys.stderr)
    assert len(volc) == len(vectors)
    vectors = np.array(vectors, dtype=np.float64)
    return (volc, vectors)

# divide X and volc into several groups
# wordGroups is a list of words set, should be disjointed
# XList: list of X
# volcList: list of volc for that X
# mapList: list of new-to-old(original) mapping
def divideXAndVolc(X, volc, wordGroups):
    XDict = dict()
    volcDict = dict()
    for name, wg in wordGroups.items():
        newX, newVolc, mapping = filterXAndVolc(X, volc, wg)
        volcDict[name] = newVolc
        XDict[name] = newX
    return XDict, volcDict

# filter X and volc by wordset
def filterXAndVolc(X, volc, wordSet):
    newVolc = Volc()
    wiList = list()
    for w in wordSet:
        if w in volc:
            newVolc.addWord(w)
            wiList.append(volc[w])
    newX = X[wiList]
    return newX, newVolc, wiList

# convert all new indexes (divided) to original index
def convertIndex(clusters, mapping):
    for cluster in clusters:
        for i in range(0, len(cluster)):
            cluster[i] = mapping[cluster[i]]
    return clusters

# notice: words not in word group will not be in any group
def mergeWordEachGroup(X, volc, groupThreshold, wordGroups):
    XDict, volcDict = divideXAndVolc(X, volc, wordGroups)
    clusters = list()
    for name in wordGroups.keys():
        cls = mergeWord(XDict[name], volcDict[name], groupThreshold[name])
        clusters.extend(cls)
    return clusters

# calculate all-pair cosine similarity -> select edges by fixed threshold
# -> cluster words by connected component 
def mergeWord(X, volc, threshold):
    print('Calculating distance(similarity) of each pair of word vector ...', file=sys.stderr)
    dist = pdist(X, 'cosine')
    # get list of selected edges by given method
    print('Selecting edges ...', file=sys.stderr)
    edgeList = getEdgeList(dist, len(volc), threshold) # list of (sim, (x1, x2))
    clusters = mergeEdgeList(edgeList, volc)
    print('volc size: %d -> %d' % (len(volc), len(clusters)), file=sys.stderr)
    return clusters

# get list of selected edges by given method
def getEdgeList(dist, nWords, threshold):
    edgeList = list()
    index = 0
    for i in range(0, nWords):
        for j in range(i+1, nWords):
            sim = 1. - dist[index]
            if sim > threshold:
                edgeList.append((sim, (i, j)))
            index += 1
    return edgeList

# edgeList: list of (sim, (x1, x2))
# wordIndexSet: all word indexes to be clustered
def mergeEdgeList2(edgeList, volc):
    g = nx.Graph()
    for sim, (i, j) in edgeList:
        g.add_edge(i, j)
    connected = nx.connected_components(g)
    remainWords = set(volc.volc.keys())
    clusters = list()
    for component in connected:
        cluster = convert2Word(component, volc)
        remainWords.difference_update(set(cluster))
        clusters.append(cluster)
        
    # each of word not in any connected component will be a cluster
    for w in remainWords:
        clusters.append([w])
    return clusters

def convert2Word(cluster, volc):
    return [volc.getWord(i) for i in cluster]

def eachWord2Cluster(wordSet, volc):
    clusters = list()
    for w in wordSet:
        if w in volc:
            clusters.append([w])
    return clusters

def allWord2Cluster(wordSet, volc):
    indexes = list(set([w for w in wordSet if w in volc]))
    return indexes


# print word clusters for human reading
def printWordCluster(clusters, outfile=sys.stdout):
    if type(clusters) == dict:
        for key, words in sorted(clusters.items(), key=lambda x:x[0]):
            for i, w in enumerate(words):
                if i == len(words) - 1:
                    print(w, end='', file=outfile)
                else:
                    print(w, end=',', file=outfile)
            print('', file=outfile)
    elif type(clusters) == list:
        for key, words in enumerate(clusters):
            for i, w in enumerate(words):
                if i == len(words) - 1:
                    print(w, end='', file=outfile)
                else:
                    print(w, end=',', file=outfile)
            print('', file=outfile)

# print word clusters as volcabulary file
def printWordClusterAsVolc(clusters, offset=0, outfile=sys.stdout):
    if type(clusters) == dict:
        for key, words in sorted(clusters.items(), key=lambda x:x[0]):
            for w in words:
                print(w, key+offset, sep=':', file=outfile)
    elif type(clusters) == list:
        for key, words in enumerate(clusters):
            for w in words:
                print(w, key+offset, sep=':', file=outfile)

def checkClusters(clusters):
    for i in range(0, len(clusters)):
        for j in range(i+1, len(clusters)):
            intersect = set(clusters[i]) & set(clusters[j])
            if len(intersect) > 0:
                print(intersect)
                return False
    return True


# deprecated
# edgeList: list of (sim, (x1, x2))
# wordIndexSet: all word indexes to be clustered
def mergeEdgeList(edgeList, volc):
    g = nx.Graph()
    for sim, (i, j) in edgeList:
        g.add_edge(i, j)
    
    partition = community.best_partition(g)
    components = convert2NodeList(partition)

    remainWords = set(volc.volc.keys())
    clusters = list()
    for component in components:
        cluster = convert2Word(component, volc)
        remainWords.difference_update(set(cluster))
        clusters.append(cluster)
        
    # each of word not in any connected component will be a cluster
    for w in remainWords:
        clusters.append([w])
    return clusters

def convert2NodeList(partition):
    num = len(set(partition.values()))
    components = list()
    for i in range(0, num):
        components.append([nId for nId, cId in partition.items() if cId == i])
    return components


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print('Usage:', sys.argv[0], 'WordVector(.vector) volcFile threshold SentiLexicon wordTagFile outWordClusterPrefix -tag threshold -tag threshold ...', file=sys.stderr)
        exit(-1)

    wordVectorFile = sys.argv[1]
    volcFile = sys.argv[2]
    threshold = float(sys.argv[3])
    sentiFile = sys.argv[4]
    wordTagFile = sys.argv[5]
    outFilePrefix = sys.argv[6]

    # initialization 
    volc = Volc()
    volc.load(volcFile)
    print('# allowed words:', len(volc), file=sys.stderr)
    # load word vectors
    volc, vectors = readWordVector(wordVectorFile, set(volc.volc.keys()))
    X = np.array(vectors)
    print(X.shape, '# actual words:', len(volc), file=sys.stderr)
    # one word should have only one POS tag (most frequent)
    with open(wordTagFile, 'r') as f:
        (wordTag, tagWord) = WordTag.loadWordTag(f)
    # sentiment lexicon file: for grouping words with same sentiment
    sentiDict = readSentiDict(sentiFile)

    # load configuration
    tagThreshold = dict()
    for i in range(7, len(sys.argv)):
        if sys.argv[i][0] == '-' and len(sys.argv) > i:
            tagThreshold[sys.argv[i][1:]] = float(sys.argv[i+1])
    print(tagThreshold)

    clusters = list()
    removedWords = set()
    # pos words is a set, neg words is a set, those words are removed
    posWordSet = set([w for w, s in sentiDict.items() if s > 0])
    negWordSet = set([w for w, s in sentiDict.items() if s < 0])
    clusters.append(allWord2Cluster(posWordSet, volc)) # pos cluster
    clusters.append(allWord2Cluster(negWordSet, volc)) # neg cluster
    removedWords.update(posWordSet | negWordSet)

    wordGroups = dict()
    for tag, threshold in tagThreshold.items():
        if tag not in tagWord: 
            print('There is no %s tag in corpus' % (tag), file=sys.stderr)
        wordSet = tagWord[tag] - removedWords
        if threshold >= 1.0: # each word is a cluster
            clusters.extend(eachWord2Cluster(wordSet, volc)) 
        elif threshold <= -1.0: # all words is a cluster
            clusters.extend(allWord2Cluster(wordSet, volc))
        else:
            wordGroups[tag] = wordSet
        removedWords.update(wordSet)

    cls = mergeWordEachGroup(X, volc, tagThreshold, wordGroups)
    clusters.extend(cls)
    print('Check clusters:', checkClusters(clusters))
    print('# clusters: ', len(clusters), file=sys.stderr)
    print('# words not in any cluster: ', len(set(volc.volc.keys()) - removedWords))

    with open(outFilePrefix + '.txt', 'w') as f:
        printWordCluster(clusters, outfile=f)
    with open(outFilePrefix + '.volc', 'w') as f:
        printWordClusterAsVolc(clusters, outfile=f)


