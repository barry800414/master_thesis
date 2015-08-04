
from multiprocessing.managers import BaseManager
import os, sys

class QueueManager(BaseManager):
    pass
class SendJob:
    def __init__(self):
        QueueManager.register('get_queue')
        port = 3333
        self.m = QueueManager(address=('140.112.187.33', 3333), authkey=b'barry800414')
        self.m.connect()
        self.queue = self.m.get_queue()

    def putTask(self, cmd):
        self.queue.put(cmd)

#sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)


resultFile = 'SingleMinMax20150804.csv'
rDir = './classifier/single'
#for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_POS', 'Dep_PPAll', 'Dep_FullAll', 'Dep_POSAll', 'Dep_stance']: 
for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll']: 
    for t in [3, 4, 5, 13]:
        data = 't%d_%s_df2' % (t, f)
        task = '%s_minmax' % (data)
        cmd = 'python3 ./classifier/Run.py ./feature/%s/%s.pickle 3 -outLogPickle %s/%s_log.pickle --preprocess -method minmax > %s/%s_result.csv' % (f, data, rDir, task, rDir, task)
        #print(cmd)
        #sender.putTask(cmd)
    
        cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
        print(cmd)
        os.system(cmd)
    os.system('echo "" >> %s' % (resultFile))


def genTopicRange(docNum):
    range1 = set(range(2, 11)) | set([round(docNum * 0.01 * i) for i in range(1, 10)]) | set([round(docNum * 0.1 * i) for i in range(1, 11)]) 
    range1 = sorted(list(range1 - set([1])))
    return range1

docNum = { 2: 125, 3: 739, 4: 116, 5: 128, 13: 194 }

#range1 = [0.1 * i for i in range(1, 11)] #original 0.1 ~ 1.0 x docNum
nI = 500
seedNum =  3
for t in [3, 4, 5, 13]:
    resultFile = 'SingleLDA20150723_t%d.csv' % (t)
    for feature in ['Dep_stance']:
    #for feature in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_PPAll', 'Dep_Full', 'Dep_FullAll', 'Dep_POS', 'Dep_POSAll', 'Dep_stance']:
        dataDir = './feature/%s' % feature
        resultDir = './dimReduction/single_LDA_result'
        for usingUnlabeledData in [0, 1]:
            # 2 ~ 10, 0.01~0.1 * docNum, 0.1 ~ 1 * docNum
            range1 = genTopicRange(docNum[t])
            #print(range1)
            for nT in range1:
                data = 't%d_%s_df2.pickle' % (t, feature)
                task = 't%d_%s_df2_LDA_nT%d_uX%d' % (t, feature, nT, usingUnlabeledData)
                
                cmd = 'python3 ./dimReduction/dimReduction.py %s/%s LDA %d -nTopics %d -nIter %d -outPickle %s/%s.pickle -outLogPickle %s/%s_log.pickle -run %d > %s/%s_result.csv' % (dataDir, data, usingUnlabeledData, nT, nI, resultDir, task, resultDir, task, seedNum, resultDir, task)

                #print(cmd)
                #sender.putTask(cmd)
    
                #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultDir, task, resultFile)
                #print(cmd)
                #os.system(cmd)
            #os.system('echo "" >> %s' % (resultFile))
        #os.system('echo "" >> %s' % (resultFile))
        


for t in [3, 4, 5, 13]:
    resultFile = 'SinglePCA20150722_t%d.csv' % (t)
    for feature in ['Dep_stance']:
    #for feature in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_PPAll', 'Dep_Full', 'Dep_FullAll', 'Dep_POS', 'Dep_POSAll', 'Dep_stance']:
        dataDir = './feature/%s' % feature
        resultDir = './dimReduction/single_PCA_result'
        for usingUnlabeledData in [0, 1]:
            # 2 ~ 10, 0.01~0.1 * docNum, 0.1 ~ 1 * docNum
            range1 = genTopicRange(docNum[t])
            #print(range1)
            for nC in range1:
                data = 't%d_%s_df2.pickle' % (t, feature)
                task = 't%d_%s_df2_PCA_nC%d_uX%d' % (t, feature, nC, usingUnlabeledData)
                
                cmd = 'python3 ./dimReduction/dimReduction.py %s/%s PCA %d -nComp %d -outPickle %s/%s.pickle -outLogPickle %s/%s_log.pickle -run %d > %s/%s_result.csv' % (dataDir, data, usingUnlabeledData, nC, resultDir, task, resultDir, task, seedNum, resultDir, task)

                #print(cmd)
                #sender.putTask(cmd)
    
                #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultDir, task, resultFile)
                #print(cmd)
                #print(cmd, file=sys.stderr)
                #os.system(cmd)
            #os.system('echo "" >> %s' % (resultFile))
        #os.system('echo "" >> %s' % (resultFile))


