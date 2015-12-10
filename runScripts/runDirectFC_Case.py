
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

libDir = '~/master_thesis/classifier'
vFile = '../dimReduction/WordClustering/news7852Final.vector'
fDir = '../feature'
#rDir = './featureMerge'

seedNum = 1
rDir = './featureMerge'

thresholdDict = {
    3: { 'BOW_tf': 0.47, '2Word': 0.75, '3Word': 0.75, 'Dep_PPAll': 0.73, 'Dep_FullAll': 0.75 },
    4: { 'BOW_tf': 0.39, '2Word': 0.6, '3Word': 0.61, 'Dep_PPAll': 0.65, 'Dep_FullAll': 0.75 },
    5: { 'BOW_tf': 0.28, '2Word': 0.73, '3Word': 0.72, 'Dep_PPAll': 0.37, 'Dep_FullAll': 0.65 },
    13: { 'BOW_tf': 0.54, '2Word': 0.75, '3Word': 0.75, 'Dep_PPAll': 0.53, 'Dep_FullAll': 0.71 }
}

# running direct feature merge (community detection)
oriDir = './single'
rDir3 = '../errorAnalysis'
resultFile = 'single_DirectFM_20150806.csv'
for preprocess in ['minmax']:
    for t in [3, 4, 5, 13]:
        for f in ['BOW_tf', '2Word', '3Word', 'Dep_FullAll', 'Dep_PPAll']:
            threshold = thresholdDict[t][f]
            data = 't%d_%s_df2' % (t, f)
            task = '%s_%s_T%g' % (data, preprocess, threshold)
            pFile = '%s/%s/%s.pickle' % (fDir, f, data)
            adjFile = '%s/%s_T%g.adjList' % (rDir, data, threshold)
            logFile = '%s/%s_log.pickle' % (rDir3, task)
            cmd = 'cd %s; python3 RunWithDirectFC.py %s %s %d -outLogPickle %s' % (libDir, pFile, adjFile, seedNum, logFile)
            if preprocess == 'minmax':
                cmd += ' --preprocess -method minmax'  
            else:
                cmd += ''

            #print(cmd)
            #sender.putTask(cmd)

            #oriTask = '%s_%s' % (data, preprocess)
            #oriLogFile = '%s/%s_log.pickle' % (oriDir, oriTask)
            #cmd = 'cd %s; python3 ../errorAnalysis/CompareFeature.py %s %s' % (libDir, oriLogFile, logFile)
            #print(oriTask, end=',')
            #sys.stdout.flush()
            #os.system(cmd)



for preprocess in ['minmax']:
    for t in [3, 4, 5, 13]:
        #for f in ['BOW_tf', '2Word', '3Word', 'Dep_FullAll', 'Dep_PPAll']:
        for f in ['Dep_FullAll']:
            for t2 in range(25, 75):
                threshold = float(t2) / 100
                #threshold = thresholdDict[t][f]
                data = 't%d_%s_df2' % (t, f)
                task = '%s_%s_T%g' % (data, preprocess, threshold)
                pFile = '%s/%s/%s.pickle' % (fDir, f, data)
                adjFile = '%s/%s_T%g.adjList' % (rDir, data, threshold)
                logFile = '%s/%s_log.pickle' % (rDir3, task)
                cmd = 'cd %s; python3 RunWithDirectFC.py %s %s %d -outLogPickle %s' % (libDir, pFile, adjFile, seedNum, logFile)
                if preprocess == 'minmax':
                    cmd += ' --preprocess -method minmax'  
                else:
                    cmd += ''

                #print(cmd)
                #sender.putTask(cmd)

                oriTask = '%s_%s' % (data, preprocess)
                oriLogFile = '%s/%s_log.pickle' % (oriDir, oriTask)
                cmd = 'cd %s; python3 ../errorAnalysis/CompareFeature.py %s %s' % (libDir, oriLogFile, logFile)
                print(oriTask, threshold, sep=',', end=',')
                sys.stdout.flush()
                os.system(cmd)

