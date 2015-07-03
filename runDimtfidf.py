from multiprocessing.managers import BaseManager
import os

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

docNum = { 2: 125, 3: 739, 4: 116, 5: 128, 13: 194 }

#range1 = [0.1 * i for i in range(1, 11)]
range1 = [0.1 * i for i in range(1, 21)]
range2 = [0.1*i for i in range(1, 11)]
feature = 'BOW_tf'
dataFolder = './feature'
resultFolder = './dimReduction/result_tfidf'

for t in [2, 3, 4, 5, 13]:
    resultFile = 'DimReduction20150703_tfidf_t%d.csv' % (t)
    for method in ['tf', 'tfidf', 'df']:
        # reduce to p% of document number
        for top in range1:
            topNum = round(docNum[t] * top)
            taskName = 't%d_%s_%sReduce_top%d' % (t, feature, method, topNum)
            cmd = 'python3 ./dimReduction/dimReduction.py %s/t%d_%s.pickle tfidf 0 -method %s -top %d -outPickle %s/%s.pickle > %s/%s_result.csv' % (
                    dataFolder, t, feature, method, topNum, resultFolder, taskName, resultFolder, taskName)
            #print(cmd)
            #sender.putTask(cmd)

            cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
            print(cmd)
            os.system(cmd)
        os.system('echo "" >> %s' % (resultFile))

        
        # reduce to p% of feature number
        for top in range2:
            topNum = top
            taskName = 't%d_%s_%sReduce_top%.2f' % (t, feature, method, topNum)
            cmd = 'python3 dimReduction.py %s/t%d_%s.pickle tfidf 0 -method %s -top %f > %s_result.csv' % (
                     dataFolder, t, feature, method, topNum, taskName)
            #print(cmd)
            #sender.putTask(cmd)
            
            cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
            print(cmd)
            os.system(cmd)
        os.system('echo "" >> %s' % (resultFile))
        
