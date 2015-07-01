

if __name__ == '__main__':
    if len(sys.argv) != :
        print('Usage:', sys.argv[0], 'pickleFile reduceMethod usingUnlabeled(0/1) -param1 value1 -param2 value2 ...', file=sys.stder)
        exit(-1)
    
    pickleFile = sys.argv[1]
    reduceMethod = sys.argv[2]
    usingUnlabeledData = True if sys.argv[3] == '1' else False
    param = dict()
    for i in range(4, len(sys.argv)):
        if sys.argv[i][0] == '-' and len(sys.argv) > i:
            key = sys.argv[i][1:]
            value = sys.argv[i+1]
            param[key] = value

    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)

    lX, ly, unX, volcDict = p['lX'], p['ly'], p['unX'], p['volcDict']
    X = vstack((lX, unX)).tocsr() if usingUnlabeledData else lX

    if reduceMethod == 'LDA':
        assert 'nTopics' in param and 'nIter' in param
        nT, nIter = float(param['nTopics']), int(param['nIter']) 
        nT = int(nT) if nT >= 1 else round(nT * X.shape[1])
        newX = runLDA(X, nTopics=nT, nIter=nIter)
        newlX = newX[0:lX.shape[0]]
        newuX = newX[lX.shape[0]:]
    elif reduceMethod == 'tfidf':
        assert 'method' in param and 'top' in param
        method, top = param['method'], float(param['top'])
        top = int(top) if top >= 1 else round(top * X.shape[1])
        newX, newVolc = reduce(X, method, top, volcDict['main'])
        
    # run cross-validation testing

    
