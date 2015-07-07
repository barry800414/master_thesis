
import sys, pickle
import numpy as np

def checkX(X):
    b = np.ones((1, X.shape[0]))
    for i in range(0, X.shape[1]):
        sum = int(b * X.getcol(i))
        if sum == 0:
            print(i)

with open(sys.argv[1], 'r+b') as f:
    p = pickle.load(f)

print('X')
checkX(p['X'])
print('unX')
checkX(p['unX'])

