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

range1 = [0.1 * i for i in range(1, 6)]
range2 = [300]
seed = 1
dataFolder = '/home/r02922010/master_thesis/data'
resultFolder = './dimReduction/LLDA_result'

#python3 LLDA.py topic /home/r02922010/master_thesis/data/t13_LLDA_data.txt seed nTopics nIters > test.csv

for t in [2, 3, 4, 5, 13]:
    resultFile = 'DimReduction20150705_LLDA_t%d.csv' % (t)
    # reduce to p% of document number
    for usingUnlabeledData in [0, 1]:
        fileSuffix = 'withUnlabel' if usingUnlabeledData else 'labelOnly'
        for nTRatio in range1:
            for nI in range2: 
                nT = round(docNum[t] * nTRatio)
                taskName = 't%d_LLDA_nT%d_nI%d_uX%d' % (t, nT, nI, usingUnlabeledData)
                cmd = 'python3 ./dimReduction/LLDA.py %s %s/t%d_LLDA_%s.txt %d %d %d %s > %s/%s_result.csv' % (
                        t, dataFolder, t, fileSuffix, seed, nT, nI, taskName, resultFolder, taskName)
                #print(cmd)
                #sender.putTask(cmd)

                #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                #print(cmd)
                #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))



# new
nTopics = [2, 3, 4, 5]
nIters = [500, 700, 900]
#alpha = [0.0001, 0.001, 0.01, 0.1, 1]
alpha = [100, 1000]
beta = [0.0001, 0.001, 0.1, 1, 10]
dnT, dIter, da, db = 3, 300, 10, 0.01  # default parameters

seed = 1
dataFolder = '/home/r02922010/master_thesis/data'
resultFolder = './dimReduction/LLDA_result'

#python3 LLDA.py topic /home/r02922010/master_thesis/data/t13_LLDA_data.txt seed nTopics nIters > test.csv
printCmd = False
sendCmd = False
collect = True
taskSet = set()
'''
            if collect: 
                cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                print(cmd)
                os.system(cmd)
        os.system('echo "" >> %s' % (resultFile))
'''

for t in [2, 3, 4, 5, 13]:
    resultFile = 'DimReduction20150705_LLDA_new_t%d.csv' % (t)
    # reduce to p% of document number
    for usingUnlabeledData in [0]:
        fileSuffix = 'withUnlabel' if usingUnlabeledData else 'labelOnly'
        nT, nI, a, b = dnT, dIter, da, db
        '''
        for nT in nTopics:
            taskSet.add((t, nT, nI, a, b))
            taskName = 't%d_LLDA_nT%d_nI%d_a%g_b%g_uX%d' % (t, nT, nI, a, b, usingUnlabeledData)
            cmd = 'python3 ./dimReduction/LLDA.py %s %s/t%d_LLDA_%s.txt %d %d %d %g %g %s %s/%s > %s/%s_result.csv' % (
                    t, dataFolder, t, fileSuffix, seed, nT, nI, a, b, taskName, resultFolder, taskName, resultFolder, taskName)
            if printCmd: print(cmd)
            if sendCmd: sender.putTask(cmd)
            if collect: 
                cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                print(cmd)
                os.system(cmd)
        os.system('echo "" >> %s' % (resultFile))
        '''
        nT, nI, a, b = dnT, dIter, da, db
        for a in alpha:
            taskSet.add((t, nT, nI, a, b))
            taskName = 't%d_LLDA_nT%d_nI%d_a%g_b%g_uX%d' % (t, nT, nI, a, b, usingUnlabeledData)
            cmd = 'python3 ./dimReduction/LLDA.py %s %s/t%d_LLDA_%s.txt %d %d %d %g %g %s %s/%s > %s/%s_result.csv' % (
                    t, dataFolder, t, fileSuffix, seed, nT, nI, a, b, taskName, resultFolder, taskName, resultFolder, taskName)
            if printCmd: print(cmd)
            if sendCmd: sender.putTask(cmd)
            if collect: 
                cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                print(cmd)
                os.system(cmd)
        if collect:
            os.system('echo "" >> %s' % (resultFile))
        '''
        nT, nI, a, b = dnT, dIter, da, db
        for b in beta:
            taskSet.add((t, nT, nI, a, b))
            taskName = 't%d_LLDA_nT%d_nI%d_a%g_b%g_uX%d' % (t, nT, nI, a, b, usingUnlabeledData)
            cmd = 'python3 ./dimReduction/LLDA.py %s %s/t%d_LLDA_%s.txt %d %d %d %g %g %s %s/%s > %s/%s_result.csv' % (
                        t, dataFolder, t, fileSuffix, seed, nT, nI, a, b, taskName, resultFolder, taskName, resultFolder, taskName)
            if printCmd: print(cmd)
            if sendCmd: sender.putTask(cmd)
            if collect: 
                cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                print(cmd)
                os.system(cmd)
        os.system('echo "" >> %s' % (resultFile))
        nT, nI, a, b = dnT, dIter, da, db
        for nI in nIters: 
            taskSet.add((t, nT, nI, a, b))
            taskName = 't%d_LLDA_nT%d_nI%d_a%g_b%g_uX%d' % (t, nT, nI, a, b, usingUnlabeledData)
            cmd = 'python3 ./dimReduction/LLDA.py %s %s/t%d_LLDA_%s.txt %d %d %d %g %g %s %s/%s > %s/%s_result.csv' % (
                    t, dataFolder, t, fileSuffix, seed, nT, nI, a, b, taskName, resultFolder, taskName, resultFolder, taskName)
            if printCmd: print(cmd)
            if sendCmd: sender.putTask(cmd)
            if collect: 
                cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                print(cmd)
                os.system(cmd)
        os.system('echo "" >> %s' % (resultFile))
        '''
        print(len(taskSet))
                #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (resultFolder, taskName, resultFile)
                #print(cmd)
                #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))



