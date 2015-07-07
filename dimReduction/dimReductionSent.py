
import sys, pickle
from scipy.sparse import vstack
from LDA import *
from tfidf import *
from RunExperiments import RunExp, ResultPrinter

# dimension reduction of sententce level feature

if __name__ == '__main__':
    if len(sys.argv) != 6 :
        print('Usage:', sys.argv[0], 'pickleFile nTopics nIter usingUnlabeled(0/1) outPickleFile', file=sys.stderr)
        exit(-1)
    
    pickleFile = sys.argv[1]
    nTopics = float(sys.argv[2])
    nIter = int(sys.argv[3])
    usingUnlabeledData = True if sys.argv[4] == '1' else False
    outPickleFile = sys.argv[5]

    with open(pickleFile, 'r+b') as f:
        p = pickle.load(f)

    # preparign data
    X = vstack((p['sentPX'], p['docPX'])).tocsr()
    unX = vstack((p['unSentPX'], p['unDocPX'])).tocsr()
    y, mainVolc = p['senty'], p['mainVolc']
    allX = vstack((X, unX)).tocsr() if usingUnlabeledData else X

    # reducing dimension 
    print('Reduction using LDA ... ', end='', file=sys.stderr)
    nT = int(nTopics) if nTopics > 1.0 else round(nTopics * X.shape[1])
    model = runLDA(allX, nTopics=nT, nIter=nIter) #the model
    newAllX = model.doc_topic_
    newVolc = mainVolc

    # if using unlabeled data, split it
    if usingUnlabeledData:
        newX, newunX = newAllX[0:X.shape[0]], newAllX[X.shape[0]:] 
    else: # otherwise transform it if there is transformer
        newX = newAllX
        newunX = model.transform(unX) if model is not None else None
    
    newSentPX, newDocPX = newX[0:p['sentPX'].shape[0]], newX[p['sentPX'].shape[0]:]
    newUnSentPX, newUnDocPX = newunX[0:p['unSentPX'].shape[0]], newunX[p['unSentPX'].shape[0]:]

    # print shape information
    print(X.shape, ' -> ', newX.shape, file=sys.stderr)
    if newunX is not None:
        print(unX.shape, ' -> ', newunX.shape, file=sys.stderr)
    else:
        print(unX.shape, ' -> None', file=sys.stderr)

    # output data 
    pObj = { 
            'docy': p['docy'], 'senty': p['senty'], 
            'docPX': newDocPX, 'unDocPX': newUnDocPX, 'sentPX': newSentPX, 'unSentPX': newUnSentPX, 
            'sentSX': p['sentSX'], 'unSentSX': p['unSentSX'], 'doc2XList': p['doc2XList'],
            'mainVolc': newVolc, 'config': p['config']
    }
    with open(outPickleFile, 'w+b') as f:
        pickle.dump(pObj, f)
    
    
