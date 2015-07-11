
import sys, pickle
import numpy as np
from scipy.io import savemat

def writeMATFile(pList, filename):
    yCell = np.zeros((len(pList), ), dtype=np.object)
    xCell = np.zeros((len(pList), ), dtype=np.object)
    for i in range(0, len(pList)):
        y = (pList[i]['y'] - 0.5) * 2.0;  # 0, 1 -> -1, 1
        yCell[i] = y.reshape((-1, 1)).astype(np.float64)
        xCell[i] = pList[i]['X'].astype(np.float64)

    savemat(filename, { 'X': xCell, 'Y': yCell })

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], 'outMatFile inPickleFile1 inPickleFile2 ...', file=sys.stderr)
        exit(-1)

    outMatFile = sys.argv[1]

    pList = list()
    for i in range(2, len(sys.argv)):
        with open(sys.argv[i], 'r+b') as f:
            pList.append(pickle.load(f))

    writeMATFile(pList, outMatFile)
