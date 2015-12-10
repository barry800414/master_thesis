
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

#sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)


seedNum = 3
foldNum = 10
libDir = '~/master_thesis/classifier/multi-task/'
rDir = '%s/result' % (libDir)
tMap = [3, 4, 5, 13]

rFile = 'mergedMultiTask_20150729.csv'
for topic in [1, 2, 3, 4]:
    #for data in ['merge1_LDA1177', 'merge2_LDA1177', 'merge1_PCA1177', 'merge2_PCA1177']:
    for data in ['merge2']:
        #for method in ['Logistic_Lasso', 'Logistic_L21', 'Logistic_Trace']:
        for method in ['Logistic_Trace']:
            task2 = '%s_%s_T%d' % (data, method, tMap[topic-1])
            for seed in range(1, seedNum+1):
                for fid in range(1, foldNum+1):
                    task = '%s_%s_T%dS%dF%d' % (data, method, tMap[topic-1], seed, fid)
                    cmd = "cd %s; matlab -nodesktop -nojvm -nosplash -r \"OneTestSingleFold('./data/%s.mat', '%s', %d, %d, %d, './result/%s'); exit\"" % (libDir, data, method, topic, seed, fid, task)
                    print(cmd)
                    sender.putTask(cmd)
        
                    #cmd = 'cat %s/%s_result.csv >> %s/%s_result.csv' % (rDir, task, rDir, task2)
                    #print(cmd)
                    #os.system(cmd)
            #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task2, rFile)
            #print(cmd)
            #os.system(cmd)
