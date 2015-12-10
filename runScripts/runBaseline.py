
from multiprocessing.managers import BaseManager
import os, sys

class QueueManager(BaseManager):
    pass
class SendJob:
    def __init__(self):
        QueueManager.register('get_queue')
        port = 3333
        self.m = QueueManager(address=('140.112.187.33', port), authkey=b'barry800414')
        self.m.connect()
        self.queue = self.m.get_queue()

    def putTask(self, cmd):
        self.queue.put(cmd)

#sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)

# minmax
dim = {
    'merge2_noPOS': [9647, 3416, 6065, 7574],
    'BOW_tf': [8326, 1909, 2832, 3296],
    '2Word': [607, 1544, 1458, 1617],
    '3Word': [1915, 613, 109, 496],
    'Dep_PP': [387, 180, 174, 151],
    'Dep_Full': [387, 103, 49, 100],
    'Dep_PPAll': [3375, 771, 537, 1147],
    'Dep_FullAll': [1669, 411, 100, 577]
}
topicList = [3, 4, 5, 13]
nI = 500
uX = 0
seedNum = 3
dataDir = './feature'
rDir = './dimReduction/FC_baseline_result'

resultFile = 'singleMerge_baseline_20150803.csv'
# LDA
for i in range(0, 4):
    t = topicList[i]
    for f in ['Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll']:
    #for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll', 'merge2_noPOS']:
        nT = dim[f][i]
        task = 't%d_%s_df2_nT%d' % (t, f, nT)
        data = '%s/t%d_%s_df2.pickle' % (f, t, f)

        cmd = 'python3 ./dimReduction/dimReduction.py %s/%s LDA 0 -nTopics %d -nIter %d -outPickle %s/%s.pickle -outLogPickle %s/%s_log.pickle -run %d > %s/%s_result.csv' % (dataDir, data, nT, nI, rDir, task, rDir, task, seedNum, rDir, task)

        #print(cmd)
        #sender.putTask(cmd)  
        #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
        #print(cmd)
        #os.system(cmd)
    #os.system('echo "" >> %s' % (resultFile))


# PCA
for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll', 'merge2_noPOS']:
    for i in range(0, 4):
        t = topicList[i]
        nC = dim[f][i]
        task = 't%d_%s_df2_nC%d' % (t, f, nC)
        data = '%s/t%d_%s_df2.pickle' % (f, t, f)

        cmd = 'python3 ./dimReduction/dimReduction.py %s/%s PCA 0 -nComp %d -outPickle %s/%s.pickle -outLogPickle %s/%s_log.pickle -run %d > %s/%s_result.csv' % (dataDir, data, nC, rDir, task, rDir, task, seedNum, rDir, task)

        #print(cmd)
        #sender.putTask(cmd)  
        
        #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
        #print(cmd)
        #os.system(cmd)
    #os.system('echo "" >> %s' % (resultFile))


### feature selection ###
def genParams(method, i, f):
    params = list()
    if method == 'chi': 
        params.append({'top': dim[f][i] })
    elif method == 'RFE':
        params.append({'n_features_to_select': dim[f][i], 'step': 0.02})
    elif method == 'RFECV':
        params.append({'step': 0.02, 'n_folds': 10, 'scorerName': 'Accuracy'})
    elif method in ['LinearSVM', 'RF']:
        params.append(None)
    return params

def genTaskName(t, feature, method, param):
    shortName = { 'n_features_to_select': 'nF', 'top': 'nf' } #nf: number of features
    skipName = set(['n_folds', 'step', 'scorerName'])
    task = 't%d_%s_df2_%s' % (t, feature, method)

    if param is None:
        return task
    for key, value in param.items():
        if key in skipName:
            continue
        else:
            task = task + '_%s%s' % (shortName[key], str(value))
    return task

seedNum =  3

fNumFile = 'fNum_20150807.csv'
for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll', 'merge2_noPOS']:
    for i in range(0, 4):
        t = topicList[i]
        for method in ['chi', 'RFE', 'RFECV', 'LinearSVM', 'RF']:
            params = genParams(method, i, f)
            for param in params:
                data = '%s/t%d_%s_df2.pickle' % (f, t, f)
                task = genTaskName(t, f, method, param)
                cmd = 'python3 ./classifier/Run.py %s/%s %d -outLogPickle %s/%s_log.pickle --fSelect -method %s' % (dataDir, data, seedNum, rDir, task, method)
                if param is not None:
                    for key, value in param.items():
                        cmd = cmd + ' -%s %s' % (key, str(value))
                cmd = cmd + ' > %s/%s_result.csv' % (rDir, task)
                
                #print(cmd)
                #sender.putTask(cmd)
            
                #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
                #print(cmd)
                #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))

                # get feature number per document
                cmd = 'python3 ./errorAnalysis/CalcFeatureNum.py log %s/%s_log.pickle >> %s' % (rDir, task, fNumFile)
                print(cmd)
                os.system(cmd)
        os.system('echo "" >> %s' % (fNumFile))

