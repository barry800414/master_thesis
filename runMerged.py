
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

sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)

for t in [3, 4, 5, 13]:
    resultFile = 'baseline20150709_t%d.csv' % (t)
    for f in ['baseline', 'baseline2']:
        cmd = 'python3 ./classifier/Run.py ./feature/%s/t%d_%s_df2.pickle 3 > ./classifier/baseline/t%d_%s_df2_results.csv' % (f, t, f, t, f)
        #print(cmd)
        #sender.putTask(cmd)
        #os.system(cmd)
    
        cmd = 'python3 CollectResult.py ./classifier/baseline/t%d_%s_df2_results.csv >> %s' % (t, f, resultFile)
        print(cmd)
        os.system(cmd)
    os.system('echo "" >> %s' % (resultFile))


def genTopicRange(docNum):
    range1 = set(range(2, 11)) | set([round(docNum * 0.01 * i) for i in range(1, 10)]) | set([round(docNum * 0.1 * i) for i in range(1, 4)]) 
    range1 = sorted(list(range1 - set([1])))
    return range1

docNum = { 2: 125, 3: 739, 4: 116, 5: 128, 13: 194 }

#range1 = [0.1 * i for i in range(1, 11)] #original 0.1 ~ 1.0 x docNum
range2 = [500]
seedNum =  3
for t in [3, 4, 5, 13]:
    resultFile = 'baseline20150709_t%d.csv' % (t)
    for feature in ['baseline', 'baseline2']:
        dataFolder = './feature/%s' % feature
        resultFolder = './dimReduction/baseline_LDA_result'
        for usingUnlabeledData in [0, 1]:
            # 2 ~ 10, 0.01~0.1 * docNum, 0.1 ~ 1 * docNum
            range1 = genTopicRange(docNum[t])
            #print(range1)
            for nT in range1:
                for nI in range2: 
                    taskName = 't%d_%s_df2_LDA_nT%d_nI%d_uX%d' % (t, feature, nT, nI, usingUnlabeledData)
                    if nT > docNum[t] or nT == 0: 
                        print('nT: ', nT)
                        continue
                    cmd = 'python3 ./dimReduction/dimReduction.py %s/t%d_%s_df2.pickle LDA %d -nTopics %d -nIter %d -outPickle %s/%s.pickle -run %d > %s/%s_result.csv' % (dataFolder, t, feature, usingUnlabeledData, nT, nI, resultFolder, taskName, seedNum, resultFolder, taskName)

                    #print(cmd)
                    #sender.putTask(cmd)
    
                    cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                    print(cmd)
                    os.system(cmd)
            os.system('echo "" >> %s' % (resultFile))


