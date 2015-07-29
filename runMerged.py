
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

sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)

### raw feature ###
rDir = './classifier/merged_raw'
resultFile = 'mergedRaw20150727.csv'
for t in [3, 4, 5, 13]:
    for f in ['merge1', 'merge2']:
        cmd = 'python3 ./classifier/Run.py ./feature/%s/t%d_%s_df2.pickle 3 %s/t%d_%s_df2_log.pickle > %s/t%d_%s_df2_results.csv' % (f, t, f, rDir, t, f, rDir, t, f)
        #print(cmd)
        #sender.putTask(cmd)
    
        #cmd = 'python3 CollectResult.py %s/t%d_%s_df2_results.csv >> %s' % (rDir, t, f, resultFile)
        #print(cmd)
        #os.system(cmd)
    #os.system('echo "" >> %s' % (resultFile))



### LDA ###
def genTopicRange(docNum):
    range1 = set(range(2, 11)) | set([round(docNum * 0.01 * i) for i in range(1, 10)]) | set([round(docNum * 0.1 * i) for i in range(1, 11)]) 
    range1 = sorted(list(range1 - set([1])))
    return range1

docNum = { 2: 125, 3: 739, 4: 116, 5: 128, 13: 194 }
nI = 500
seedNum =  3
for uX in [0, 1]:
    for t in [3, 4, 5, 13]:
        resultFile = 'mergedLDA_20150727_t%d.csv' % (t)
        for feature in ['merge1', 'merge2']:
            dataDir = './feature/%s' % feature
            rDir = './dimReduction/merged_LDA_result'
            # 2 ~ 10, 0.01~0.1 * docNum, 0.1 ~ 1 * docNum
            range1 = genTopicRange(docNum[t])
            #print(range1)
            for nT in range1:
                task = 't%d_%s_df2_LDA_nT%d_uX%d' % (t, feature, nT, uX)
                cmd = 'python3 ./dimReduction/dimReduction.py %s/t%d_%s_df2.pickle LDA %d -nTopics %d -nIter %d -outPickle %s/%s.pickle -outLogPickle %s/%s_log.pickle -run %d > %s/%s_result.csv' % (dataDir, t, feature, uX, nT, nI, rDir, task, rDir, task, seedNum, rDir, task)

                #print(cmd)
                #sender.putTask(cmd)

                #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
                #print(cmd)
                #os.system(cmd)
            #os.system('echo "" >> %s' % (resultFile))

### PCA ###
seedNum =  3
uX = 0
for t in [3, 4, 5, 13]:
    resultFile = 'mergedPCA_20150727_t%d.csv' % (t)
    for feature in ['merge1', 'merge2']:
        dataDir = './feature/%s' % feature
        rDir = './dimReduction/merged_PCA_result'
        
        # 2 ~ 10, 0.01~0.1 * docNum, 0.1 ~ 1 * docNum
        range1 = genTopicRange(docNum[t])
        #print(range1)
        for nC in range1:
            task = 't%d_%s_df2_PCA_nT%d_uX%d' % (t, feature, nC, uX)
            cmd = 'python3 ./dimReduction/dimReduction.py %s/t%d_%s_df2.pickle PCA %d -nComp %d -outPickle %s/%s.pickle -outLogPickle %s/%s_log.pickle -run %d > %s/%s_result.csv' % (dataDir, t, feature, uX, nC, rDir, task, rDir, task, seedNum, rDir, task)

            #print(cmd)
            #sender.putTask(cmd)

            #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
            #print(cmd)
            #os.system(cmd)
    #os.system('echo "" >> %s' % (resultFile))



### feature selection ###
def genParams(method, docNum):
    params = list()
    if method == 'chi': 
        range1 = genTopicRange(docNum)
        for num in range1:
            params.append({'top': num })
    elif method == 'RFE':
        range1 = genTopicRange(docNum)
        for num in range1:
            params.append({'n_features_to_select': num, 'step': 0.02})
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
rDir = './classifier/merged_fSelect_result'
for t in [3, 4, 5, 13]:
    resultFile = 'merged_fSelect_20150729_t%d.csv' % (t)
    for feature in ['merge1', 'merge2']:
        dataDir = './feature/%s' % feature
        #for method in ['chi', 'RFE', 'RFECV', 'LinearSVM', 'RF']:
        for method in ['RFECV']:
            params = genParams(method, docNum[t])
            for param in params:
                task = genTaskName(t, feature, method, param)
                cmd = 'python3 ./classifier/Run.py %s/t%d_%s_df2.pickle %d -outLogPickle %s/%s_log.pickle --fSelect -method %s' % (dataDir, t, feature, seedNum, rDir, task, method)
                if param is not None:
                    for key, value in param.items():
                        cmd = cmd + ' -%s %s' % (key, str(value))
                cmd = cmd + ' > %s/%s_result.csv' % (rDir, task)
                
                print(cmd)
                sender.putTask(cmd)

                #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
                #print(cmd)
                #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))


### allTrainOneTest - raw ###
rDir = './classifier/merged_raw'
resultFile = 'mergedRaw_DirectTrain_20150729.csv'
for t in [3, 4, 5, 13]:
    for f in ['merge1_shareVolc', 'merge2_shareVolc']:
        task = 't%d_%s_DirectTrain' % (t, f)
        cmd = 'python3 ./classifier/OneTestRun.py %d 3' % (t)
        for i in [3, 4, 5, 13]:
            cmd += ' -%d ./feature/%s/t%s_%s.pickle' % (i, f, i, f)
        cmd += ' -outLogPickle %s/%s_log.pickle > %s/%s_result.csv' % (rDir, task, rDir, task)
        #print(cmd)
        #sender.putTask(cmd)
        
        #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
        #print(cmd)
        #os.system(cmd)
    #os.system('echo "" >> %s' % (resultFile))

### allTrainOneTest - LDA & PCA ###
rDir = './classifier/merged_raw'
resultFile = 'mergedPCALDA_DirectTrain_20150729.csv'
for t in [3, 4, 5, 13]:
    for f in ['merge1_shareVolc', 'merge2_shareVolc']:
        for d in ['PCA1177', 'LDA1177']:
            task = 't%d_%s_%s_DirectTrain' % (t, f, d)
            cmd = 'python3 ./classifier/OneTestRun.py %d 3' % (t)
            for i in [3, 4, 5, 13]:
                cmd += ' -%d ./feature/%s/t%s_%s_%s.pickle' % (i, f, i, f, d)
            cmd += ' -outLogPickle %s/%s_log.pickle > %s/%s_result.csv' % (rDir, task, rDir, task)
            #print(cmd)
            #sender.putTask(cmd)
        
            #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
            #print(cmd)
            #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))

