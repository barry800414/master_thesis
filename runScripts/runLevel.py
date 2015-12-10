
from multiprocessing.managers import BaseManager
import os, sys

port = 3333
print('port:', port, file=sys.stderr)
class QueueManager(BaseManager):
    pass
class SendJob:
    def __init__(self):
        QueueManager.register('get_queue')
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


seedNum = 3
libFolder = '~/master_thesis/classifier/multi-level2/'
resultPrefix = 'MultiLevel_BOW_20150717'
dataName = { 13: 't13_BOW_tf_nT16_ux0.mat', 3:'t3_BOW_tf_nT22_ux0.mat', 4:'t4_BOW_tf_nT9_ux0.mat', 5:'t5_BOW_tf_nT13_ux0.mat' }
learn_rate = 0.01
#updateDict = { 'Least': ['GD', 'ALS'], 'Logistic': ['GD'] }
updateDict = { 'Least': ['ALS'] }
maxIterDict = { 'GD': [100, 200, 300], 'ALS': [3, 5, 10, 20] }

for t in [3, 4, 5, 13]:
    for method in ['Least']:
        for update in updateDict[method]:
            for maxIter in maxIterDict[update]:
                taskName = '%s_%s_nI%d' % (method, update, maxIter)
                cmd = "cd %s; matlab -nodesktop -nojvm -nosplash -r \"RunTask('./data/%s', '%s', %f, %d, '%s', %d, 1, './result/MultiLevel_t%d_%s'); exit\"" % (libFolder, dataName[t], method, learn_rate, maxIter, update, seedNum, t, taskName)

                #print(cmd)
            #sender.putTask(cmd)
            
            #cmd = 'python3 clean.py ./classifier/multi-task/result/MultTask_%s_%s_result.csv' % (taskName, method)

learn_rate = 0.01
updateDict = { 'Least': ['GD', 'ALS'], 'Logistic': ['GD'] }
maxIterDict = { 'GD': [300], 'ALS': [3] }
stopList = ['trainFunc', 'valFunc', 'valErr', 'ValAcc']
foldNum = 10
resultDir = './classifier/multi-level2/result/'
resultFile = 'MultiLevel_LDA_20150717.csv'
for t in [3, 4, 5, 13]:
    for method in ['Least', 'Logistic']:
        for update in updateDict[method]:
            for maxIter in maxIterDict[update]:
                for stop in stopList:
                    for seed in range(1, seedNum+1):
                        for fid in range(1, foldNum+1):
                            taskName = '%s_%s_%s_nI%d_S%dF%d' % (method, stop, update, maxIter, seed, fid)
                            cmd = "cd %s; matlab -nodesktop -nojvm -nosplash -r \"RunTaskSingleFold('./data/%s', '%s', '%s', '%s', %f, %d, %d, %d, 1, './result/MultiLevel_t%d_%s'); exit\"" % (libFolder, dataName[t], method, update, stop, learn_rate, maxIter, seed, fid, t, taskName)

                            #print(cmd)
                            #sender.putTask(cmd)
                    
                            #cmd = 'cat %s/MultiLevel_t%d_%s_result.csv >> %s/MultiLevel_t%d_%s_%s_nI%d_result.csv' % (resultDir, t, taskName, resultDir, t, method, update, maxIter)
                            #print(cmd)
                            #os.system(cmd)
                    #cmd = 'python3 CollectResult.py %s/MultiLevel_t%d_%s_%s_nI%d_result.csv >> %s' % (resultDir, t, method, update, maxIter, resultFile)
                    #print(cmd)
                    #os.system(cmd)


learn_rate = 0.01
foldNum = 10
resultDir = './classifier/multi-level2/result'
resultFile = 'MultiLevel_BOW_20150718.csv'
dataName = { 13: 't13_BOW_tf.mat', 3:'t3_BOW_tf.mat', 4:'t4_BOW_tf.mat', 5:'t5_BOW_tf.mat' }
maxIterDict = { 'GD': [300], 'ALS': [3] }
updateDict = { 'Least': ['ALS'], 'Logistic': ['GD'] }
#updateDict = { 'Least': ['GD', 'ALS'], 'Logistic': ['GD'] }
#stopList = ['trainFunc', 'valFunc', 'valErr', 'valAcc']
stopList = ['valAcc']

for t in [3, 4, 5, 13]:
    for method in ['Least']:
        for update in updateDict[method]:
            for maxIter in maxIterDict[update]:
                for stop in stopList:
                    taskName2 = 'BOW_tf_%s_%s_%s_nI%d' % (method, update, stop, maxIter)
                    for seed in range(1, seedNum+1):
                        for fid in range(1, foldNum+1):
                            taskName = 'BOW_tf_%s_%s_%s_nI%d_S%dF%d' % (method, update, stop, maxIter, seed, fid)
                            cmd = "cd %s; matlab -nodesktop -nojvm -nosplash -r \"RunTaskSingleFold('./data/%s', '%s', '%s', '%s', %f, %d, %d, %d, 1, './result/MultiLevel_t%d_%s'); exit\"" % (libFolder, dataName[t], method, update, stop, learn_rate, maxIter, seed, fid, t, taskName)

                            #print(cmd)
                            #sender.putTask(cmd)
                            
                            #cmd = 'cat %s/MultiLevel_t%d_%s_result.csv >> %s/MultiLevel_t%d_%s_result.csv' % (resultDir, t, taskName, resultDir, t, taskName2)
                            #print(cmd)
                            #os.system(cmd)
                    #cmd = 'python3 CollectResult.py %s/MultiLevel_t%d_%s_result.csv >> %s' % (resultDir, t, taskName2, resultFile)
                    #print(cmd)
                    #os.system(cmd)


learn_rate = 0.01
foldNum = 10
resultDir = './classifier/multi-level2/result'
resultFile = 'MultiLevel_BOW_PCALDA_20150719.csv'
dataType = ['BOW_sent_tf_PCA1.0', 'BOW_sent_tf_LDA1']
update = 'GD'
maxIter = 300
stop = 'valFunc'

for t in [3, 4, 5, 13]:
    for method in ['Logistic', 'Least']:
        for data in dataType:
            taskName2 = '%s_%s_%s_nI%d' % (data, method, update, maxIter)
            dataName = 't%d_%s.mat' % (t, data)
            for seed in range(1, seedNum+1):
                for fid in range(1, foldNum+1):
                    taskName = '%s_S%dF%d' % (taskName2, seed, fid)
                    cmd = "cd %s; matlab -nodesktop -nojvm -nosplash -r \"RunTaskSingleFoldLineSearch('./data/%s', '%s', '%s', '%s', %f, %d, %d, %d, 1, './result/MultiLevel_t%d_%s'); exit\"" % (libFolder, dataName, method, update, stop, learn_rate, maxIter, seed, fid, t, taskName)

                    #print(cmd)
                    #sender.putTask(cmd)
                    
                    #cmd = 'cat %s/MultiLevel_t%d_%s_result.csv >> %s/MultiLevel_t%d_%s_result.csv' % (resultDir, t, taskName, resultDir, t, taskName2)
                    #print(cmd)
                    #os.system(cmd)
            cmd = 'python3 CollectResult.py %s/MultiLevel_t%d_%s_result.csv >> %s' % (resultDir, t, taskName2, resultFile)
            print(cmd)
            os.system(cmd)
            os.system('echo "" >> %s' % (resultFile))
