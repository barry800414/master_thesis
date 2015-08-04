
import sys, math, pickle, copy
from FeatureMerge import *


def saveFeatureVector(vectors, filename):
    with open(filename, 'w') as f:
        for vector in vectors:
            if vector is None:
                print('', file=f)
            else:
                for i, v in enumerate(vector):
                    if i == len(vector) - 1:
                        print(v, end='\n', file=f)
                    else:
                        print(v, end=' ', file=f)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:', sys.argv[0], 'WordVector(.vector) pickleFile outFile', file=sys.stderr)
        exit(-1)

    wordVectorFile = sys.argv[1]
    pickleFile = sys.argv[2]
    outFile = sys.argv[3]

    # read word vectors
    volc, vectors = readWordVector(wordVectorFile)
    wordVector = toDictType(volc, vectors)

    # read pickle file (volc)
    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)
    volc = p['mainVolc']
    print('# feature:', len(volc), file=sys.stderr)

    vectors = getFeatureVectors_noSkip(volc, wordVector)
    assert len(vectors) == len(volc)
    
    saveFeatureVector(vectors, outFile)
