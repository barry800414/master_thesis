
import sys, os, pickle
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from plotModel import *


libFolder = '/home/r02922010/master_thesis/dimReduction/socialMF'

def readYFile(filename):
    y = list()
    with open(filename, 'r') as f:
        for line in f:
            y.append(int(line.strip()))
    return np.array(y)

def saveIgnoreDoc(filename, testIndex, yLength):
    with open(filename, 'w') as f:
        print(yLength, file=f)
        for i in testIndex:
            print(i, file=f)

def readLatentFile(filename, k):
    userLatent = list()
    itemLatent = list()
    now = userLatent
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                now = itemLatent
            else:
                entry = line.split()
                assert len(entry) == k
                latent = list()
                for e in entry:
                    latent.append(float(e))
                now.append(latent)
    return (np.array(userLatent), np.array(itemLatent))

def osRunSRMF(taskName, XFile, edgeListFile, testIndex, yLength, k, uReg, iReg, maxIter, uSocialReg, seed):
    ignoreFile = '%s_ignoreDoc.txt' % (taskName)
    saveIgnoreDoc(ignoreFile, testIndex, yLength)
    modelFile = '%s.model' % (taskName)
    cmd = '%s/sr_mf %s 1 %s -k %d -uReg %f -iReg %f -iter %d -userNetwork %s %f -seed %d -ignoreUserIdFile %s' % (
            libFolder, XFile, modelFile, k, uReg, iReg, maxIter, edgeListFile, uSocialReg, seed, ignoreFile)
    #cmd = '%s/sr_mf %s 1 %s -k %d -uReg %f -iReg %f -iter %d -seed %d -ignoreUserIdFile %s' % (
    #        libFolder, XFile, modelFile, k, uReg, iReg, maxIter, seed, ignoreFile)
    os.system(cmd)
    print(cmd)
    (userLatent, itemLatent) = readLatentFile(modelFile, k)
    os.system('rm %s' % (modelFile))
    return userLatent

if __name__ == '__main__':
    if len(sys.argv) != 12:
        print('Usage:', sys.argv[0], 'taskName XFile EdgeListFile yFile k uReg iReg maxIter uSocialReg seed nFolds', file=sys.stderr)
        exit(-1);

    taskName = sys.argv[1]
    XFile = sys.argv[2]
    edgeListFile = sys.argv[3]
    yFile = sys.argv[4]
    k = int(sys.argv[5])
    uReg = float(sys.argv[6])
    iReg = float(sys.argv[7])
    maxIter = float(sys.argv[8])
    uSocialReg = float(sys.argv[9])
    seed = int(sys.argv[10])
    nFolds = int(sys.argv[11])

    y = readYFile(yFile)
   
    for fid, (trainIndex, testIndex) in enumerate(StratifiedKFold(y, n_folds=nFolds)):
        newName = '%s_k%d_u%g_i%g_iter%d_us%g_s%d_fold%d' % (taskName, k, uReg, uReg, maxIter, uSocialReg, seed, fid)
        X = osRunSRMF(newName, XFile, edgeListFile, testIndex, len(y), k, uReg, iReg, maxIter, uSocialReg, seed)
        plotModel(X, y, trainIndex, testIndex)
        break
