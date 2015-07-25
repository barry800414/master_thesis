
import sys, pickle
from Volc import Volc

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'inPickle outVolc', file=sys.stderr)
        exit(-1)

    with open(sys.argv[1], 'r+b') as f:
       p =  pickle.load(f)

    volc = p['mainVolc'] 
    volc.save(sys.argv[2])
