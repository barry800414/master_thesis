
# Whole framework to do two-side feature merging, using community detection

import sys, math, random, copy
from collections import defaultdict

import numpy as np
from numpy.matrixlib.defmatrix import matrix
from scipy.sparse import csr_matrix, hstack, vstack

from sklearn import svm, grid_search
from sklearn.cross_validation import StratifiedKFold 
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, make_scorer
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE, RFECV, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone

# need sklearn > 0.16.0
from sklearn.cross_validation import PredefinedSplit


from misc import *
import FeatureMerge as FM


# class for providing frameworks for running experiments
class RunExp:
    def selfTrainTestNFoldWithFC(version, X, y, volc, adjSet, clfName, scorerName, randSeed=1, test_folds=10,
            cv_folds=10, fSelectConfig=None, preprocess=None, n_jobs=-1, outfile=sys.stdout):
        # check data
        if not DataTool.XyIsValid(X, y): #do nothing
            return
        
        # split data in N-fold
        skf = StratifiedKFold(y, n_folds=test_folds, shuffle=True, random_state=randSeed)
        logs = list()
        for fId, (trainIndex, testIndex) in enumerate(skf):
            XTrain, XTest = X[trainIndex], X[testIndex]
            yTrain, yTest = y[trainIndex], y[testIndex]

            # training using cross-validation and feature clustering 
            # version1:
            # version2: 10-fold -> train first time on val -> feature merge -> grid search -> train first on test -> feature merge
            if version == 1:
                print('Running version 1...', file=sys.stderr)
                (clf, bestParam, trainScore, yTrainPredict, valScore, model, newXTest) = ML.GridSearchCVandTrainWithFC(
                        XTrain, yTrain, XTest, preprocess, volc, adjSet, clfName, scorerName, cv_folds, randSeed, n_jobs=n_jobs)
            elif version == 2:
                print('Running version 2...', file=sys.stderr)
                (clf, bestParam, trainScore, yTrainPredict, valScore, model, newXTest) = ML.GridSearchCVandTrainWithFC_v2(
                        XTrain, yTrain, XTest, preprocess, volc, adjSet, clfName, scorerName, cv_folds, randSeed, n_jobs=n_jobs)
            #TODO: to dump the model (how features are merged) to see the physical meaning 
            if XTest is None or yTest is None:
                predict = { 'yTrainPredict': yTrainPredict }
            else:
                # testing 
                # yTestPredict = ML.testWithFC(newXTest, model, clf)
                yTestPredict = clf.predict(newXTest)
                # evaluation
                testScore = Evaluator.evaluate(yTestPredict, yTest)
                predict = { 'yTrainPredict': yTrainPredict, 'yTestPredict': yTestPredict } 
            
            # to save information of training process for further analysis
            log = { 'clfName': clfName, 'clf': clf, 'param': bestParam, 'trainScore': trainScore, 'valScore': valScore,
                    'predict': predict, 'testScore': testScore, 'split':{ 'trainIndex': trainIndex, 'testIndex': testIndex}, 
                    'model': model }
            if fSelectConfig is not None:
                log['XTrain'] = XTrain
                log['XTest'] = XTest

            logs.append(log)
            ResultPrinter.print('SelfTrainTestWithFC', clfName, scorerName, model.toDim(), 
                    randSeed, fId, trainScore, valScore, testScore[scorerName], outfile=outfile)

        return logs

    # only a portion of data in test topic is testing data, all other data (including in
    # other topic) is training data
    def allTrainOneTestNFold(X, y, topicMap, topic, clfName, scorerName, randSeed=1, test_folds=10,
            cv_folds=10, fSelectConfig=None, prefix='', outfile=sys.stdout):
        # check data
        if not DataTool.XyIsValid(X, y): return #do nothing
        # making scorer
        scorer = Evaluator.makeScorer(scorerName)

        # split data
        skf = DataTool.OneTestStratifiedKFold(y, topicMap, topic, randSeed, test_folds)

        logs = list()
        for fId, (trainIndex, testIndex) in enumerate(skf):
            XTrain, XTest = X[trainIndex], X[testIndex]
            yTrain, yTest = y[trainIndex], y[testIndex]
            # do feature selection if config is given
            if fSelectIsBeforeClf(fSelectConfig) == True:
                print('before selection:', XTrain.shape, file=sys.stderr)
                (XTrain, selector) = ML.fSelect(XTrain, yTrain, fSelectConfig['method'], 
                        fSelectConfig['params'])
                print('after selection:', XTrain.shape, file=sys.stderr)
                if XTest is not None:
                    XTest = selector.transform(XTest)

            # training using validation
            trainTopicMap = [topicMap[i] for i in trainIndex]
            (clf, bestParam, trainScore, valScore, yTrainPredict) = ML.OneTestTrain(XTrain, 
                    yTrain, trainTopicMap, topic, clfName, scorer, randSeed, n_folds=cv_folds)

            if XTest is None or yTest is None:
                predict = { 'yTrainPredict': yTrainPredict }
            else:
                # testing 
                yTestPredict = ML.test(XTest, clf)
                # evaluation
                testScore = Evaluator.evaluate(yTestPredict, yTest)
                predict = { 'yTrainPredict': yTrainPredict, 'yTestPredict': yTestPredict } 
            
            # to save information of training process for further analysis
            log = { 'clfName': clfName, 'clf': clf, 'param': bestParam, 'trainScore': trainScore, 'valScore': valScore,
                    'predict': predict, 'testScore': testScore, 'split':{ 'trainIndex': trainIndex, 'testIndex': testIndex} }
            logs.append(log)
            ResultPrinter.print('SelfTrainTest', clfName, scorerName, X.shape[1], 
                    randSeed, fId, trainScore, valScore, testScore[scorerName], outfile=outfile)

        return logs


# The class for providing functions to manipulate data
class DataTool:
    # In order to prevent 0 cases in training or testing data (s.t. evaluation 
    # metric is illed-defined), we first get the expected number of instance first
    def stratifiedSplitTrainTest(X, y, randSeed=1, testSize=0.2, originalIndex = None):
        assert X.shape[0] == len(y)
        assert testSize >= 0.0 and testSize < 1.0
        #print('testSize:', testSize)
        
        if testSize == 0.0:
            return (X, None, y, None)
        length = len(y)

        # count the number of instance for each y class
        yNum = defaultdict(int)
        for i in range(0, length):
            yNum[y[i]] += 1

        # calculate expected number of instance for each class
        yTestNum = { yi: int(math.ceil(cnt*testSize)) for yi, cnt in yNum.items() }
        yTrainNum = { yi: yNum[yi] - yTestNum[yi] for yi in yNum.keys() }
        
        # random shuffle
        index = [i for i in range(0, length)]
        random.seed(randSeed)
        random.shuffle(index)

        nowNum = { yi: 0 for yi in yNum.keys() }
        trainIndex = list()
        testIndex = list()
        for i in index:
            if nowNum[y[i]] < yTrainNum[y[i]]:
                trainIndex.append(i)
                nowNum[y[i]] += 1
            else:
                testIndex.append(i)
        XTrain = X[trainIndex]
        XTest = X[testIndex]
        yTrain = y[trainIndex]
        yTest = y[testIndex]
        
        # if original index (of X) is given, then mapping trainIndex 
        # and testIndex into original index
        if originalIndex is not None:
            trainIndex = [originalIndex[i] for i in trainIndex]
            testIndex = [originalIndex[i] for i in testIndex]

        #print('y:', y)
        #print('yTrain:', yTrain)
        #print('yTest:', yTest)
        
        return (XTrain, XTest, yTrain, yTest, trainIndex, testIndex)


    # only the data in test topic will be viewed as testing data
    # topic: the target testing topic
    def OneTestStratifiedKFold(y, topicMap, topic, randSeed, n_folds):
        topicy = list() # y in that topic
        topicIndexList = list() # the list of original index of testing data
        otherIndexList = list()
        for i, t in enumerate(topicMap):
            if t == topic:
                topicy.append(y[i])
                topicIndexList.append(i)
            else:
                otherIndexList.append(i)

        topicy = np.array(topicy)
        kfold = StratifiedKFold(topicy, n_folds=n_folds, shuffle=True, random_state=randSeed)
        
        newKFold = list() # generating new whole k fold 
        for i, (topicTrainIndex, topicTestIndex) in enumerate(kfold):
            trainIndex = [topicIndexList[j] for j in topicTrainIndex]
            trainIndex.extend(otherIndexList) # all other data are training data
            testIndex = [topicIndexList[j] for j in topicTestIndex]
            newKFold.append((trainIndex, testIndex))


        testSet = set()
        for i, (trainIndex, testIndex) in enumerate(newKFold):
            if len(set(trainIndex) & set(testIndex)) != 0:
                print('Error!!!!!!!!!!!!!!!!!!!!!!!!!!', file=sys.stderr)
            if len(testSet & set(testIndex)) != 0:
                print('Error2 !!!!!!!!!!!!!!!!!!!!!!!!!', file=sys.stderr)
            testSet.update(set(testIndex))

        return newKFold

    # horzontally merge two matrix, height should be identical
    def hstack(X1, X2):
        if X1.shape[0] != X2.shape[0]:
            print('X1%s and X2%s has different height' % (X1.shape, X2.shape), file=sys.stderr)
            return None
        if type(X1) != type(X2):
            print('X1(%s) and X2(%s) are different type of matrix' % (type(X1), type(X2)), file=sys.stderr)
            newX1 = X1.toarray() if type(X1) == csr_matrix else X1
            newX2 = X2.toarray() if type(X2) == csr_matrix else X2
            return np.concatenate((newX1, newX2), axis=1) 

        # concatenate XTrain Matrix
        xType = type(X1)
        if xType == matrix: #dense
            return np.concatenate((X1, X2), axis=1)
        elif xType == csr_matrix: #sparse
            return csr_matrix(hstack((X1, X2)))
        
    def XyIsValid(X, y):
        if X.shape[0] != len(y):
            print('X.shape[0] != len(y)', file=sys.stderr)
            return False
        elif X.shape[0] == 0:
            print('X.shape[0] == 0', file=sys.stderr)
            return False
        elif X.shape[1] == 0:
            print('X.shape[1] == 0', file=sys.stderr)
            return False
        else:
            return True
    
    def saveAsLibSVMFormat(X, y, outfile=sys.stdout):
        assert X.shape[0] == len(y)
        if type(X) == csr_matrix:
            (rowNum, colNum) = X.shape
            colIndex = X.indices
            rowPtr = X.indptr
            data = X.data
            nowPos = 0
    
            sumOfCol = [0.0 for i in range(0, colNum)]
            for ri in range(0, rowNum):
                print(y[ri], end='', file=outfile)
                for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
                    value = data[nowPos]
                    print(' %d:%f' % (ci, value), end='', file=outfile)
                    nowPos += 1
                print('', file=outfile)
    
    # preprocessing the feature matrix (including standardization, 
    # minMax scaling, normalization and binarization)
    def preprocessX(X, method, params):
        success = False
        # standardization: transforming each column(feature) to zero-mean, std=1
        if method in ['std', 'standardization']:
            print('Using standardization ...', file=sys.stderr)
            scaler = preprocessing.StandardScaler(with_mean=params['with_mean'], 
                    with_std=params['with_std']).fit(X)
            preX = scaler.transform(X)
            success = True
        # minMax scaling: transforming each column(feature) to [a, b]
        # (usually [-1, 1] or [0, 1])
        elif method in ['minmax', 'minMax']:
            if 'feature_range' not in params or params['feature_range'] is None:
                feature_range = (0, 1)
            else:
                feature_range = params['feature_range']
            print('Using MinMax scaling to', feature_range, file=sys.stderr)
            preX = DataTool.minMaxScaling(X, feature_range)
            success = True
        # normalization: transforming scaling each row(an instance) to fixed length 
        # (usually L1-norm or L2-norm to length 1)
        elif method in ['norm', 'normalization']:
            if 'norm' in params:
                print('Using normalization (%s)' % (params['norm']), file=sys.stderr)
                preX = preprocessing.normalize(X, norm=params['norm'])
                success = True
        # binarization: transforming each entry to 0 or 1 by given threshold
        elif method in ['0/1', 'binary', 'binarization']:
            if 'threshold' in params:
                binarizer = preprocessing.Binarizer(threshold=params['threshold'])
                preX = binarizer.transform(X)
                success = True
        else:
            print('Preprocessing method not found', file=sys.stderr)
        
        if not success:
            print('preprocessing X failed', file=sys.stderr)
            assert 1 == 0

        return preX

    # min max scaling for sparse matrix
    # if matrix has negative value, then most of 0 value will not be changed
    def minMaxScaling(X, feature_range=(0.0, 1.0)):
        (rowNum, colNum) = X.shape
        colIndex = X.indices
        rowPtr = X.indptr
        data = X.data

        # traverse whole matrix to get min and max of the column
        nowPos = 0
        minOfCol = [0.0 for i in range(0, colNum)]
        maxOfCol = [0.0 for i in range(0, colNum)]
        for ri in range(0, rowNum):
            for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
                v = data[nowPos]
                if v < minOfCol[ci]:
                    minOfCol[ci] = float(v)
                if v > maxOfCol[ci]:
                    maxOfCol[ci] = float(v)
                nowPos += 1

        #print(minOfCol)
        #print(maxOfCol)
        nowPos = 0
        interval = [maxOfCol[i] - minOfCol[i] for i in range(0, colNum)]
        #print(interval)
        minV, maxV = feature_range
        for ri in range(0, rowNum):
            for ci in colIndex[rowPtr[ri]:rowPtr[ri+1]]:
                if not float_eq(interval[ci], 0.0):
                    v = data[nowPos]
                    v_std = (v - minOfCol[ci]) / interval[ci]
                    v_scaled = v_std * (maxV - minV) + minV
                    #print(v, v_std, v_scaled)
                    data[nowPos] = v_scaled
                nowPos += 1
        return X


def trainTestWithFC_OneTask(XTrain, yTrain, XTest, yTest, clfName, clfParams, scorerName):
    return ML.trainTestWithFC(XTrain, yTrain, XTest, yTest, clfName, clfParams, scorerName)

def trainFirstandFC_OneTask(XTrain, yTrain, XTest, preprocess, volc, adjSet):
    return ML.trainFirstandFC(XTrain, yTrain, XTest, preprocess, volc, adjSet)

# The class for providing function to do machine learning procedure
class ML:
    # version1
    def GridSearchCVandTrainWithFC(XTrain, yTrain, XTest, preprocess, volc, 
            adjSet, clfName, scorerName, n_folds, randSeed, n_jobs=-1):

        # train first to do feature clustering
        newXTrain, newXTest, model = ML.trainFirstandFC(XTrain, yTrain, XTest, preprocess, volc, adjSet)

        # find best parameters
        (clf, bestParam, trainScore, yTrainPredict, valScore) = ML.GridSearchCVandTrain(
                newXTrain, yTrain, clfName, scorerName, n_folds, randSeed, n_jobs)
        
        return (clf, bestParam, trainScore, yTrainPredict, valScore, model, newXTest)

    # version1
    def GridSearchCVandTrain(X, y, clfName, scorerName, n_folds, randSeed=1, n_jobs=-1):
        # make cross validation iterator 
        #print(' n_folds:', n_folds, end='', file=sys.stderr) 
        kfold = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=randSeed)
        return ML.__GridSearchCVandTrain(kfold, X, y, clfName, scorerName, n_jobs)

    # version1
    def __GridSearchCVandTrain(kfold, X, y, clfName, scorerName, n_jobs=-1):
        scorer = Evaluator.makeScorer(scorerName)
        # get classifier and parameters to try
        (clf, parameters) = ML.genClfAndParams(clfName)

        # get grid search classifier
        #print('->grid search ', end='', file=sys.stderr)
        clfGS = GridSearchCV(clf, parameters, scoring=scorer, refit=True, cv=kfold, n_jobs=n_jobs)
        
        # refit the data by the best parameters
        clfGS.fit(X, y)

        # get validation score
        bestValScore = clfGS.best_score_

        # testing on training data
        predict = clfGS.predict(X)
        trainScore = Evaluator.score(scorerName, y, predict)

        return (clfGS.best_estimator_, clfGS.best_params_, trainScore, predict, bestValScore)



    def testWithFC(X, model, clf):
        # train first time with default classifier
        newX = model.transform(X)
        predict = clf.predict(newX)
        return predict

    # version2
    # 10-fold -> train first time on val -> feature merge -> grid search -> train first on test -> feature merge
    def GridSearchCVandTrainWithFC_v2(XTrain, yTrain, XTest, preprocess, volc, adjSet,
            clfName, scorerName, n_folds, randSeed, n_jobs=-1):
        # find best parameters
        (valScore, bestParams) = ML.GridSearchCVWithFC_v2(XTrain, yTrain, 
                preprocess, volc, adjSet, clfName, scorerName, n_folds, randSeed, n_jobs)
        
        # train first to do feature clustering
        newXTrain, newXTest, model = ML.trainFirstandFC(XTrain, yTrain, XTest, 
                preprocess, volc, adjSet)

        # refit on training data by best parameters
        (clf, yTrainPredict, trainScore) = ML.trainWithFC(newXTrain, 
                yTrain, clfName, bestParams, scorerName)

        return (clf, bestParams, trainScore, yTrainPredict, valScore, model, newXTest)

    # version2
    def GridSearchCVWithFC_v2(X, y, preprocess, volc, adjSet, clfName, 
            scorerName, n_folds, randSeed, n_jobs):
   
        kfold = StratifiedKFold(y, n_folds=n_folds, random_state=randSeed)
        (clf, parameters) = ML.genClfAndParams(clfName)
        paramsGrid = ParameterGrid(parameters)

        # train first time to do feature clustering for each fold
        out = Parallel(n_jobs=n_jobs)(delayed(trainFirstandFC_OneTask)(
            X[train], y[train], X[test], preprocess, volc, adjSet)
                for k, (train, test) in enumerate(kfold))
        XTrainList = [out[i][0] for i in range(0, n_folds)]
        XTestList = [out[i][1] for i in range(0, n_folds)]

        # grid search to find best parameters
        out = Parallel(n_jobs=n_jobs)(delayed(trainTestWithFC_OneTask)(
            XTrainList[k], y[train], XTestList[k], y[test], clfName, params, scorerName) 
                for params in paramsGrid 
                for k, (train, test) in enumerate(kfold))
        
        bestParams = None
        bestScore = None
        n_fits = len(out)
        # collecting results
        for grid_start in range(0, n_fits, n_folds):
            avgScore = 0.0
            for r in out[grid_start:grid_start + n_folds]:
                avgScore += r['score']
            avgScore /= n_folds
            if bestScore is None or avgScore > bestScore:
                bestScore = avgScore
                bestParams = out[grid_start]['params']
        
        return (bestScore, bestParams)

    # version1 and version2 will use it
    def trainFirstandFC(XTrain, yTrain, XTest, preprocess, volc, adjSet):
        # train first time with default classifier
        #print('Training using default classifier ...', file=sys.stderr)
        clf = ML.genClf(None)
        clf.fit(XTrain, yTrain)

        # feature clustering #TODO: dump the model to see the physical meaning 
        model = FM.featureClustering(clf.coef_, volc, adjSet)

        # merge train & test -> remove df < 2 & preprocess -> split X
        X = vstack((XTrain, XTest)).tocsr()
        X1 = model.transform(X)
        DF = countDFByCSRMatrix(X1)
        X2 = shrinkCSRMatrixByDF(X1, DF, minCnt=2)
        print(X.shape, '--feature merge-->', X1.shape, '--remove df<2-->', X2.shape, file=sys.stderr)
        if preprocess is not None:
            X2 = DataTool.preprocessX(X2, preprocess['method'], preprocess['params'])
            print(X2)
        newXTrain = X2[0:XTrain.shape[0]]
        newXTest = X2[XTrain.shape[0]:]
        print('Train:', XTrain.shape, '->', newXTrain.shape, ' Test:', XTest.shape, '->', newXTest.shape, file=sys.stderr)
        return newXTrain, newXTest, model

    # version2 
    def trainTestWithFC(XTrain, yTrain, XTest, yTest, clfName, clfParams, scorerName):
        (clf, yTrainPredict, trainScore) = ML.trainWithFC(XTrain, yTrain, clfName, clfParams, scorerName)
        yTestPredict = clf.predict(XTest)
        testScore = Evaluator.score(scorerName, yTest, yTestPredict)
        return { 'score': testScore, 'params': clfParams }

    # version2
    def trainWithFC(X, y, clfName, clfParams, scorerName):
        clf = ML.genClf({'clfName': clfName, 'params': clfParams})
        clf.fit(X, y)
        predict = clf.predict(X)
        score = Evaluator.score(scorerName, y, predict)
        return (clf, predict, score)

    def __train(kfold, XTrain, yTrain, clfName, scorer, n_folds, randSeed=1, n_jobs=-1):
        # get classifier and parameters to try
        (clf, parameters) = ML.genClfAndParams(clfName)

        # get grid search classifier
        #print('->grid search ', end='', file=sys.stderr)
        clfGS = GridSearchCV(clf, parameters, scoring=scorer, refit=True, cv=kfold, n_jobs=n_jobs)
        
        # refit the data by the best parameters
        #print('->refit ', end='', file=sys.stderr)
        clfGS.fit(XTrain, yTrain)

        # get validation score
        bestValScore = clfGS.best_score_

        # testing on training data
        yPredict = clfGS.predict(XTrain)
        if scorer == 'accuracy':
            trainScore = accuracy_score(yTrain, yPredict)
        else:
            trainScore = scorer.score(yTrain, yPredict)

        return (clfGS.best_estimator_, clfGS.best_params_, trainScore, bestValScore, yPredict)

    # new
    def OneTestTrain(XTrain, yTrain, topicMap, topic, clfName, scorer, randSeed, n_folds, n_jobs=-1):
        kfold = DataTool.OneTestStratifiedKFold(yTrain, topicMap, topic, randSeed, n_folds) 
        return ML.__train(kfold, XTrain, yTrain, clfName, scorer, n_folds, randSeed, n_jobs)

        # get classifier and parameters to try
        (clf, parameters) = ML.genClfAndParams(clfName)
        
        (bestValScore, bestParams) = ML.OneTestGridSearchCV(clf, parameters, 
                topicMap, topic, scorerName, XTrain, yTrain, n_folds, randSeed, n_jobs=-1)
        
        # refit the data by the best parameters
        clf.set_params(**bestParams)
        #print('-> topic refit ', end='', file=sys.stderr)
        clf.fit(XTrain, yTrain)
        
        # testing on training data
        yPredict = clf.predict(XTrain)
        
        return (clf, bestParams, bestValScore, yPredict)

    def test(XTest, bestClf):
        yPredict = bestClf.predict(XTest)
        return yPredict

    def genClfAndParams(clfName):
        if clfName == 'NaiveBayes':
            parameters = {
                'alpha': [0.5, 1.0, 2.0],
                'fit_prior': [True, False]
            }
            clf = MultinomialNB()
        elif clfName == 'MaxEnt' or clfName == 'LogReg':
            C = [math.pow(2, i) for i in range(-15,5,1)]
            parameters = {
                'penalty': ['l2',],
                'C': C,
                'class_weight': ['balanced'],
                'dual': [True, False],
                }
            #parameters = {
            #    'penalty': ['l1',],
            #    'C': C,
            #    'class_weight': ['balanced'],
            #    'dual': [False],
            #    }

            clf = LogisticRegression()
        elif clfName == 'SVM':
            C = [math.pow(2, i) for i in range(-15,5,2)]
            gamma = [math.pow(2, i) for i in range(-11,-1,2)]
            parameters = {
                    'kernel': ('rbf', ), 
                    'C': C, 
                    'gamma': gamma
                }
            clf = svm.SVC()
        elif clfName == 'LinearSVM' or clfName == 'LinearSVC':
            C = [math.pow(2, i) for i in range(-19,5,2)]
            parameters = {
                    'C': C,
                    'class_weight': ['balanced'],
                    'loss': ['squared_hinge'],
                    'dual': [True, False],
                    #'penalty': ['l1', 'l2']
                }
            clf = svm.LinearSVC()
        elif clfName == 'RandomForest' or clfName == 'RF': #depricated: RF does not support sparse matrix
            estNum = [5, 10, 20, 40, 80]
            minSampleSplit = [1, 2]
            parameters = {
                    "n_estimators": estNum,
                    "min_samples_split": minSampleSplit,
                    "class_weight": ['balanced']
                }
            clf = RandomForestClassifier()
        else:
            print('Classifier name cannot be identitified', file=sys.stderr)
            return None

        return (clf, parameters) 

    def genClf(config):
        if config == None:
            return LogisticRegression()
        clfName = config['clfName']
        params = config['params']
        if clfName == 'NaiveBayes':
            clf = MultinomialNB(**params)
        elif clfName == 'MaxEnt' or clfName == 'LogReg':
            clf = LogisticRegression(**params)
        elif clfName == 'SVM':
            clf = svm.SVC(**params)
        elif clfName == 'LinearSVM' or clfName == 'LinearSVC':
            clf = svm.LinearSVC(**params)
        elif clfName == 'RandomForest' or clfName == 'RF': #depricated: RF does not support sparse matrix
            clf = RandomForestClassifier(**params)
        else:
            print('Classifier name cannot be identitified', file=sys.stderr)
            return None
        return clf

    # only applicable for SelfTrainTest ?
    
    #def genCV(config, y):
    #    cvType = config['cvType']
    #    params = config['params']
    #    if cvType in ['kFold', 'KFold']:
    #        cv = StratifiedKFold(y, **params)
    #    elif cvType in ['LOO', 'LeaveOneTest']:
    #        cv = LeaveOneTest(len(y), **params) 
    #    else:
    #        print('CV Type cannot be found', file=sys.stderr)
    #    return cv

    def genCV(n_folds, y):
        return StratifiedKFold(y, n_folds=n_folds)

    # feature selection 
    # FIXME: for the method using classifier, the target score is Accuracy (rather than MacroF1)
    def fSelect(XTrain, yTrain, method, params):
        if method in ["chi", "chi-square"]:
            assert 'top' in params
            top = params['top']
            if top > 1.0:
                print('Selecting features with top %d chi-square value ...' % top, file=sys.stderr) 
                selector = SelectKBest(chi2, k=top).fit(XTrain, yTrain)
                newX = selector.transform(XTrain)
            else:
                top = int(top * 100)
                print('Selecting %d%% features with top chi-square value ...' % top, file=sys.stderr)
                selector = SelectPercentile(chi2, percentile=top).fit(XTrain, yTrain)
                newX = selector.transform(XTrain)

        elif method in ["rfe", "RFE", 'RecursiveFeatureElimination']:
            assert 'n_features_to_select' in params and 'step' in params
            print('Selecting features using RecursiveFeatureElimination ...', file=sys.stderr)
            clf = ML.genClf(params['clfConfig']) if 'clfConfig' in params else ML.genClf(None)
            selector = RFE(clf, params['n_features_to_select'], step=params['step']).fit(XTrain, yTrain)
            newX = selector.transform(XTrain)

        elif method in ["rfecv", "RFECV"]:  # only applicable to SelfTrainTest
            if 'n_folds' in params and 'step' in params:
                print('Selecting features using RecursiveFeatureElimination and CrossValidation ... n_folds:', params['n_folds'], file=sys.stderr)
                clf = ML.genClf(params['clfConfig']) if 'clfConfig' in params else ML.genClf(None)
                cv = ML.genCV(params['n_folds'], yTrain) 
                scorer = Evaluator.makeScorer(params['scorerName'])
                selector = RFECV(clf, step=params['step'], cv=cv, scoring=scorer).fit(XTrain, yTrain)
                newX = selector.transform(XTrain)

        elif method in ['LinearSVM', 'LinearSVC']:
            C = params['C'] if 'C' in params else 1.0
            print('Selecting features using Linear SVM (L1 regularizaton) ... C:', C, file=sys.stderr)
            selector = svm.LinearSVC(C=C, penalty="l1", dual=False).fit(XTrain, yTrain)
            newX = selector.transform(XTrain)

        elif method in ['rf', 'RF', 'RandomForest']:
            print('Selecting features using RandomForest ...', file=sys.stderr)
            selector = RandomForestClassifier().fit(XTrain, yTrain)
            newX = selector.transform(XTrain)

        # the meaning of volcabulary will be missing
        elif method in ['LDA', 'LinearDiscriminantAnalysis']:
            print('dimension reduction using LinearDiscriminantAnalysis', file=sys.stderr)
            if 'n_components' in params and 'solver' in params and 'shrinkage' in params:
                selector = discriminant_analysis.LinearDiscriminantAnalysis(**params).fit(XTrain.todense(), yTrain)
                newX = selector.transform(XTrain.todense())
        return (newX, selector)

def fSelectIsBeforeClf(fSelectConfig):
    if fSelectConfig == None or 'method' not in fSelectConfig:
        return None
    method = fSelectConfig['method']
    if method in ["chi", "chi-square", 'LinearSVM', 'LinearSVC', 'rf', 'RF', 'RandomForest', 
            "rfe", "RFE", 'RecursiveFeatureElimination', 'rfecv', 'RFECV', 'LDA']:
        return True
    else:
        return None

def macroF1Score(yTrue, yPredict):
    f1List = f1_score(yTrue, yPredict, average=None)
    return np.sum(f1List) / len(f1List)


macroF1Scorer = make_scorer(macroF1Score)
macroRScorer = make_scorer(recall_score, average='macro')
accScorer = make_scorer(accuracy_score)
scorerMap = {"Accuracy" : accScorer, "MacroF1": macroF1Scorer, "MacroR": macroRScorer } 


# The class for providing function to evaluate results
class Evaluator:
    def score(scorerName, yTrue, yPredict):
        if scorerName in ['accuracy', 'Accuracy']:
            return accuracy_score(yTrue, yPredict)

    def evaluate(yPredict, yTrue):
        # accuracy 
        accu = accuracy_score(yTrue, yPredict)
        # f1 score for each class (not binary)
        f1List = f1_score(yTrue, yPredict, average=None)
        # macroF1
        macroF1 = sum(f1List) / len(f1List)
        # average recall (macro recall)
        #recall = recall_score(yTrue, yPredict, average='macro')
        result = { 'Accuracy': accu, 'MacroF1': macroF1 }
        for i, f1 in enumerate(f1List):
            name = 'F1_' + i2Label[i]
            result[name] = f1
        return result
    
    def avgNFoldResult(resultList):
        avgScore = defaultdict(float)
        for result in resultList:
            for key, value in result.items():
                avgScore[key] += value
        for key in avgScore.keys():
            avgScore[key] = avgScore[key] / len(resultList)
        return avgScore

    def avgTopicResults(topicResults, weights):
        if topicResults is None or len(topicResults) == 0:
            return None
        cnt = 0
        avgScore = defaultdict(float)
        for t, r in topicResults.items():
            for scorerName in r.keys():
                avgScore[scorerName] += weights[t] * r[scorerName]
        
        return dict(avgScore)

    def makeScorer(scorerName, topicMap=None):
        if topicMap is None:
            return scorerMap[scorerName]
        else:
            return make_scorer(Evaluator.topicMacroF1Scorer, 
                    topicMap=topicMap, greater_is_bette=True)

    def topicMacroF1Scorer(yTrue, yPredict, **kwargs):
        assert 'topicMap' in set(kwargs.keys())
        topicMap = kwargs['topicMap']
        (topicResult, avgR) = Evaluator.topicEvaluate(yPredict, yTrue, topicMap)
        return avgR['MacroF1']

class ResultPrinter:
    def printFirstLine(outfile=sys.stdout):
        print('framework, classifier, scorer, dimension, randSeed, foldNum, train, val, test', file=outfile)

    def getDataType():
        return ('str', 'str', 'str', 'str', 'int', 'int', 'float', 'float', 'float')

    def getDataType2():
        return ('str', 'str', 'str', 'str', 'int', 'int', 'float', 'float', 'float', 
                'float', 'float', 'float', 
                'float', 'float', 'float', 
                'float', 'float', 'float', 
                'float', 'float', 'float')

    def print(f, clf, s, d, seed, fid, train, val, test, outfile=sys.stdout):
        print(f, clf, s, d, seed, fid, train, val, test, sep=',',file=outfile)
