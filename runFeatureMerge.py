
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
for t in [3, 4, 5, 13]:
    #for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll']:
    for f in ['Dep_PP', 'Dep_PPAll']:
    #for f in ['Dep_PPAll']:
        #data = 't%d_%s_df2' % (t, f)
        data = 't%d_%s_df2' % (t, f)
        pFile = '%s/%s/%s.pickle' % (fDir, f, data)
        outPrefix = '%s/%s' % (rDir, data)
        cmd = 'cd %s; python3 FeatureMerge.py %s %s %s %d' % (libDir, vFile, pFile, outPrefix, v)
        #for t2 in range(25, 66, 1):
        for t2 in range(66, 76, 1):
            threshold = float(t2) / 100
            cmd += ' %g' % (threshold)
        #print(cmd)
        #sender.putTask(cmd)

# run feature merging (community detection) for single feature
resultFile = 'single_FM_20150802.csv'
preprocess = 'minmax'
v = 2
rDir2 = './merged_FM_result_v%d' % (v) 
for t in [3, 4, 5, 13]:
    for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll']:
    #for f in ['merge2_noPOS']:
        #for t2 in range(25, 66, 1):
        for t2 in range(66, 76, 1):
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
            
            #cmd = 'cd %s; python3 Pickle2CSV.py %s %s' % (libDir, logFile, rFile)
            #print(cmd)
            #os.system(cmd)
            
            #cmd = 'python3 CollectResult.py %s/%s >> %s' % (libDir, rFile, resultFile)
            #print(cmd)
            #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))

# run feature merging (community detection) for merged feature

thresholdDict = {
    3: { 'BOW_tf': 0.4, '2Word': 0.4, '3Word': 0.4, 'Dep_PPAll': 0.4, 'Dep_FullAll': 0.4 },
    4: { 'BOW_tf': 0.4, '2Word': 0.4, '3Word': 0.4, 'Dep_PPAll': 0.4, 'Dep_FullAll': 0.4 },
    5: { 'BOW_tf': 0.4, '2Word': 0.4, '3Word': 0.4, 'Dep_PPAll': 0.4, 'Dep_FullAll': 0.4 },
    13: { 'BOW_tf': 0.4, '2Word': 0.4, '3Word': 0.4, 'Dep_PPAll': 0.4, 'Dep_FullAll': 0.4 }
}

preprocess = 'minmax'
resultFile = 'merge_FM_20150802.csv'
for t in [3, 4, 5, 13]:
    for tDiff in range(-10, 11, 1):
        task = 'merge_%s_Diff%g' % (preprocess, tDiff)
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
        print(cmd)
        #sender.putTask(cmd)
            
            #cmd = 'cd %s; python3 Pickle2CSV.py %s %s' % (libDir, logFile, rFile)
            #print(cmd)
            #os.system(cmd)
            
            #cmd = 'python3 CollectResult.py %s/%s >> %s' % (libDir, rFile, resultFile)
            #print(cmd)
            #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))



# running direct feature merge (community detection)
rDir2 = './merged_DirectFM_result'
resultFile = 'singleMerge_DirectFM_20150803.csv'
for preprocess in ['minmax', '']:
    for t in [3, 4, 5, 13]:
        for f in ['BOW_tf', '2Word', '3Word', 'Dep_Full', 'Dep_PP', 'Dep_FullAll', 'Dep_PPAll', 'merge2_noPOS']:
        #for f in ['BOW_tf']:
        #for f in ['merge2_noPOS']:
        #for f in ['Dep_PPAll']:
            for t2 in range(25, 66, 1):
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
                
                #cmd = 'cd %s; python3 Pickle2CSV.py %s %s' % (libDir, logFile, rFile)
                #print(cmd)
                #os.system(cmd)
                
                #cmd = 'python3 CollectResult.py %s/%s >> %s' % (libDir, rFile, resultFile)
                #print(cmd)
                #os.system(cmd)
            #os.system('echo "" >> %s' % (resultFile))

