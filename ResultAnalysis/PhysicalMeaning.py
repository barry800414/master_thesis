
import sys, pickle


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'PickleFile OutputPrefix', file=sys.stderr)
        exit(-1)
        
    with open(sys.argv[1], 'r+b') as f:
        p = pickle.load(f)
    
    print(p[0].keys())
    
