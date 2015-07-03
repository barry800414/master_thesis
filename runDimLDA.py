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

range1 = [0.1 * i for i in range(1, 11)]
range2 = [500]
feature = 'BOW_tf'
dataFolder = './feature'
resultFolder = './dimReduction/LDA_result'

for t in [2, 3, 4, 5, 13]:
    resultFile = 'DimReduction20150702_LDA_t%d.csv' % (t)
    # reduce to p% of document number
    for usingUnlabeledData in [0, 1]:
        for nT in range1:
            for nI in range2: 
                nT = round(docNum[t] * nT)
                taskName = 't%d_%s_LDA_nT%d_nI%d_uX%d' % (t, feature, nT, nI, usingUnlabeledData)
                cmd = 'python3 ./dimReduction/dimReduction.py %s/t%d_%s_dfmin5.pickle LDA %d -nTopics %d -nIter %d -outPickle %s/%s.pickle > %s/%s_result.csv' % (
                        dataFolder, t, feature, usingUnlabeledData, nT, nI, resultFolder, taskName, resultFolder, taskName)
                #print(cmd)
                #sender.putTask(cmd)

                cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                print(cmd)
                os.system(cmd)
        os.system('echo "" >> %s' % (resultFile))


