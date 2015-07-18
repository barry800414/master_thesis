
import sys, pickle
import numpy as np
from scipy.io import savemat

# convert sentence level data to matlab format

def writeMATFile(p, filename):
    nDocs = len(p['docy'])
    doc2XList = p['doc2XList']
    sentSX = p['sentSX']
    sentPX = p['sentPX']
    F_sta = np.zeros((nDocs, ), dtype=np.object)
    F_subj = np.zeros((nDocs, ), dtype=np.object)
    
    for i in range(0, nDocs):
        index = doc2XList[i]
        F_sta[i] = sentPX[index].astype(np.float64)
        F_subj[i] = sentSX[index].astype(np.float64)
        
    y = (p['docy'] - 0.5) * 2.0;
    y = y.reshape((-1, 1)).astype(np.float64)
    savemat(filename, { 'F_sta': F_sta, 'F_subj': F_subj, 'y': y })

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'inPickleFile outMatFile', file=sys.stderr)
        exit(-1)

    inPickleFile = sys.argv[1]
    outMatFile = sys.argv[2]

    with open(inPickleFile, 'r+b') as f:
        p = pickle.load(f)

    writeMATFile(p, outMatFile)
