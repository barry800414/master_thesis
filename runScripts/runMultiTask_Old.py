
from multiprocessing.managers import BaseManager
import os, sys

port = 3334
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
libFolder = '~/master_thesis/classifier/multi-task/'
nTRange = [2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60]

resultPrefix = 'MultiTask_BOW_20150710'
tMap = [3, 4, 5, 13]
for usingUnlabeledData in [0, 1]:
    for nT in nTRange:
        for method in ['Least_Lasso', 'Logistic_Lasso', 'Least_L21', 'Logistic_L21', 'Least_Trace', 'Logistic_Trace']:
            taskName = 'LDA_nT%d_uX%d' % (nT, usingUnlabeledData)
            cmd = "cd %s; matlab -nodesktop -nojvm -nosplash -r \"RunTask('./data/%s.mat', '%s', %d, './result/MultTask_%s_%s'); exit\"" % (libFolder, taskName, method, seedNum, taskName, method)
            #print(cmd)
            #sender.putTask(cmd)
            
            #cmd = 'python3 clean.py ./classifier/multi-task/result/MultTask_%s_%s_result.csv' % (taskName, method)

tMap = [3, 4, 5, 13]
for i in range(0, 4):
    resultFile = 'MultiTask_BOW_20150710_t%d.csv' % (tMap[i])
    for method in ['Least_Lasso', 'Logistic_Lasso', 'Least_L21', 'Logistic_L21', 'Least_Trace', 'Logistic_Trace']:
        for usingUnlabeledData in [0, 1]:
            for nT in nTRange:
                taskName = 'LDA_nT%d_uX%d' % (nT, usingUnlabeledData)
                cmd = 'python3 CollectResult.py ./classifier/multi-task/result/MultTask_%s_%s_result.csv %d >> %s' % (taskName, method, i+1, resultFile)
                #print(cmd)
                #os.system(cmd)
            #os.system('echo "" >> %s' % (resultFile))

resDir = './classifier/multi-task/result'
for i in range(0, 4):
    resultFile = 'MultiTask_BOW_20150713_t%d.csv' % (tMap[i])
    for runType in ["", "2"]:
        #for method in ['Least_Lasso', 'Logistic_Lasso', 'Least_L21', 'Logistic_L21', 'Least_Trace', 'Logistic_Trace',  'Least_Dirty', 'Least_SparseTrace', 'Least_CASO', 'Logistic_CASO']:
        for method in ['Least_Lasso', 'Logistic_Lasso', 'Least_L21', 'Logistic_L21', 'Least_Trace', 'Logistic_Trace',  'Least_Dirty', 'Least_SparseTrace']:
            taskName = 'BOW_tf_df2_PCANone'
            cmd = "cd %s; matlab -nodesktop -nojvm -nosplash -r \"RunTask%s('./data/%s.mat', '%s', %d, './result/MultTask_%s_%s_Run%s'); exit\"" % (libFolder, runType, taskName, method, seedNum, taskName, method, runType)
            #print(cmd)
            #sender.putTask(cmd)

            cmd = 'python3 CollectResult.py %s/MultTask_%s_%s_Run%s_result.csv %d >> %s' % (resDir, taskName, method, runType, i+1, resultFile)
            print(cmd)
            os.system(cmd)

# new data
for runType in ["", "2"]:
    #for method in ['Least_Lasso', 'Logistic_Lasso', 'Least_L21', 'Logistic_L21', 'Least_Trace', 'Logistic_Trace',  'Least_Dirty', 'Least_SparseTrace']:
    #for method in ['Least_Lasso', 'Logistic_Lasso', 'Least_L21', 'Logistic_L21', 'Least_Trace', 'Logistic_Trace']:
    for method in ['Logistic_Trace']:
        for seed in [1, 2, 3]:
            for fi in range(1, 11):
                taskName = 'v2'
                cmd = "cd %s; matlab -nodesktop -nojvm -nosplash -r \"RunTask%sSingleFold('./data/%s.mat', '%s', %d, %d, './result/MultTask_%s_%s_Run%s_S%dF%d'); exit\"" % (libFolder, runType, taskName, method, seed, fi, taskName, method, runType, seed, fi)
                #print(cmd)
                #sender.putTask(cmd)
