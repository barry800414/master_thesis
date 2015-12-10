
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

#sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)

libDir = '~/master_thesis/classifier'
vFile = '../dimReduction/WordClustering/news7852Final.vector'
fDir = '../feature'
#rDir = './featureMerge'

seedNum = 3
rDir = './featureMerge'
v = 1
# now 0.25~0.90
#for t in [3, 4, 5, 13]:
for t in [5]:
    #for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll']:
    for f in ['BOW_tf']:
        #data = 't%d_%s_df2' % (t, f)
        data = 't%d_%s_df2' % (t, f)
        pFile = '%s/%s/%s.pickle' % (fDir, f, data)
        outPrefix = '%s/%s' % (rDir, data)
        cmd = 'cd %s; python3 FeatureMerge.py %s %s %s %d' % (libDir, vFile, pFile, outPrefix, v)
        #for t2 in range(25, 66, 1):
        for t2 in range(20, 26, 1):
            threshold = float(t2) / 100
            cmd += ' %g' % (threshold)
        #print(cmd)
        #sender.putTask(cmd)

# run feature merging (community detection) for single feature
resultFile = 'single_FM_20150806.csv'
preprocess = 'minmax'
v = 2
rDir2 = './merged_FM_result_v%d' % (v) 
for t in [3, 4, 5, 13]:
    for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll']:
        for t2 in range(25, 76, 1):
            threshold = float(t2) / 100
            data = 't%d_%s_df2' % (t, f)
            task = '%s_%s_T%g' % (data, preprocess, threshold)
            pFile = '%s/%s/%s.pickle' % (fDir, f, data)
            adjFile = '%s/%s_T%g.adjList' % (rDir, data, threshold)
            logFile = '%s/%s_log.pickle' % (rDir2, task)
            rFile = '%s/%s_result.csv' % (rDir2, task)
            cmd = 'cd %s; python3 RunWithFC.py %s %s %d %d -outLogPickle %s' % (libDir, pFile, adjFile, v, seedNum, logFile)
            if preprocess == 'minmax':
                cmd += ' --preprocess -method minmax > %s' % (rFile) 
            else:
                cmd += ' > %s' % (rFile)

            #print(cmd)
            #sender.putTask(cmd)
            
            #cmd = 'python3 CollectResult.py %s/%s >> %s' % (libDir, rFile, resultFile)
            #print(cmd)
            #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))

# run feature merging (community detection) for merged feature
thresholdDict = {
    3: { 'BOW_tf': 0.74, '2Word': 0.75, '3Word': 0.69, 'Dep_PPAll': 0.55, 'Dep_FullAll': 0.75 },
    4: { 'BOW_tf': 0.33, '2Word': 0.75, '3Word': 0.68, 'Dep_PPAll': 0.32, 'Dep_FullAll': 0.69 },
    5: { 'BOW_tf': 0.72, '2Word': 0.74, '3Word': 0.39, 'Dep_PPAll': 0.29, 'Dep_FullAll': 0.75 },
    13: { 'BOW_tf': 0.66, '2Word': 0.64, '3Word': 0.59, 'Dep_PPAll': 0.72, 'Dep_FullAll': 0.71 }
}

preprocess = 'minmax'
resultFile = 'merge_FM_20150807.csv'
for t in [3, 4, 5, 13]:
    for tDiff in range(-10, 11, 1):
        task = 't%d_merge_%s_Diff%g' % (t, preprocess, tDiff)
        logFile = '%s/%s_log.pickle' % (rDir2, task)
        rFile = '%s/%s_result.csv' % (rDir2, task)
        cmd = 'cd %s; python3 RunWithFC_Multi.py %d --inFile' % (libDir, seedNum)
        for f in ['BOW_tf', '2Word', '3Word', 'Dep_PPAll', 'Dep_FullAll']:
            threshold = thresholdDict[t][f] + float(tDiff) / 100
            data = 't%d_%s_df2' % (t, f)
            pFile = '%s/%s/%s.pickle' % (fDir, f, data)
            adjFile = '%s/%s_T%g.adjList' % (rDir, data, threshold)
            cmd += ' %s %s' % (pFile, adjFile)

        if preprocess == 'minmax':
            cmd += ' -outLogPickle %s --preprocess -method minmax > %s' % (logFile, rFile) 
        else:
            cmd += ' -outLogPickle %s > %s' % (logFile, rFile)
        #print(cmd)
        #sender.putTask(cmd)
            
        #cmd = 'python3 CollectResult.py %s/%s >> %s' % (libDir, rFile, resultFile)
        #print(cmd)
        #os.system(cmd)
    #os.system('echo "" >> %s' % (resultFile))







# running direct feature merge (community detection)
rDir2 = './merged_DirectFM_result'
resultFile = 'single_DirectFM_20150806.csv'
for preprocess in ['minmax']:
    for t in [3, 4, 5, 13]:
        for f in ['BOW_tf', '2Word', '3Word', 'Dep_Full', 'Dep_PP', 'Dep_FullAll', 'Dep_PPAll']:
        #for f in ['merge2_noPOS']:
            for t2 in range(25, 76, 1):
                threshold = float(t2) / 100
                data = 't%d_%s_df2' % (t, f)
                task = '%s_%s_T%g' % (data, preprocess, threshold)
                pFile = '%s/%s/%s.pickle' % (fDir, f, data)
                adjFile = '%s/%s_T%g.adjList' % (rDir, data, threshold)
                logFile = '%s/%s_log.pickle' % (rDir2, task)
                rFile = '%s/%s_result.csv' % (rDir2, task)
                cmd = 'cd %s; python3 RunWithDirectFC.py %s %s %d -outLogPickle %s' % (libDir, pFile, adjFile, seedNum, logFile)
                if preprocess == 'minmax':
                    cmd += ' --preprocess -method minmax > %s' % (rFile) 
                else:
                    cmd += ' > %s' % (rFile)

                #print(cmd)
                #sender.putTask(cmd)
                
                #cmd = 'python3 CollectResult.py %s/%s >> %s' % (libDir, rFile, resultFile)
                #print(cmd)
                #os.system(cmd)
            #os.system('echo "" >> %s' % (resultFile))


# run direct feature merging (community detection) for merged feature
thresholdDict = {
    3: { 'BOW_tf': 0.47, '2Word': 0.75, '3Word': 0.75, 'Dep_PPAll': 0.73, 'Dep_FullAll': 0.75 },
    4: { 'BOW_tf': 0.39, '2Word': 0.6, '3Word': 0.61, 'Dep_PPAll': 0.65, 'Dep_FullAll': 0.75 },
    5: { 'BOW_tf': 0.28, '2Word': 0.73, '3Word': 0.72, 'Dep_PPAll': 0.37, 'Dep_FullAll': 0.65 },
    13: { 'BOW_tf': 0.54, '2Word': 0.75, '3Word': 0.75, 'Dep_PPAll': 0.53, 'Dep_FullAll': 0.71 }
}

rDir2 = './merged_DirectFM_result'
preprocess = 'minmax'
resultFile = 'merge_DirectFM_20150807.csv'
for t in [3, 4, 5, 13]:
    for tDiff in range(-10, 11, 1):
        task = 't%d_merge_%s_Diff%g' % (t, preprocess, tDiff)
        logFile = '%s/%s_log.pickle' % (rDir2, task)
        rFile = '%s/%s_result.csv' % (rDir2, task)
        cmd = 'cd %s; python3 RunWithDirectFC_Multi.py %d --inFile' % (libDir, seedNum)
        for f in ['BOW_tf', '2Word', '3Word', 'Dep_PPAll', 'Dep_FullAll']:
            threshold = thresholdDict[t][f] + float(tDiff) / 100
            data = 't%d_%s_df2' % (t, f)
            pFile = '%s/%s/%s.pickle' % (fDir, f, data)
            adjFile = '%s/%s_T%g.adjList' % (rDir, data, threshold)
            cmd += ' %s %s' % (pFile, adjFile)

        if preprocess == 'minmax':
            cmd += ' -outLogPickle %s --preprocess -method minmax > %s' % (logFile, rFile) 
        else:
            cmd += ' -outLogPickle %s > %s' % (logFile, rFile)
        #print(cmd)
        #sender.putTask(cmd)
            
        cmd = 'python3 CollectResult.py %s/%s >> %s' % (libDir, rFile, resultFile)
        print(cmd)
        os.system(cmd)
    os.system('echo "" >> %s' % (resultFile))


