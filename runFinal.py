
from multiprocessing.managers import BaseManager
import os, sys

class QueueManager(BaseManager):
    pass
class SendJob:
    def __init__(self):
        QueueManager.register('get_queue')
        port = 3333
        self.m = QueueManager(address=('140.112.31.187', port), authkey=b'barry800414')
        self.m.connect()
        self.queue = self.m.get_queue()

    def putTask(self, cmd):
        self.queue.put(cmd)

#sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)

# BOW_tf: BOW: Word
# 2Word: Bi: BiWord
# 3Word: Tri: TriWord
# PT_SB: Dep_PP: H_P, T_P, H/T_P, H_N, T_N, H/T_N
# Full_SB: Dep_Full (Full = full representation): OT, HO, HO/OT
# PT_TB: Dep_PPAll (all = TB tag based): H_P, T_P, H/T_P, H_N, T_N, H/T_N
# Full_TB: Dep_FullAll: OT, HO, HO/OT

wvFile = './classifier/news7852Final.vector'
#rDir = './featureMerge'

seedNum = 3
v = 2

### run direct feature merging (using KMeans) for single feature
resultFile = 'Single_DFM_Kmeans_20160130.csv'
rDir = './tmp_result'
cnt = 0

for t in [3, 4, 5, 13]:
    for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll']: 
        for nClusters in [i * 0.01 for i in range(25, 76)]:
            data = 't%d_%s_df2' % (t, f)
            task = '%s_c%.2f_minmax' % (data, nClusters)
            #cmd = 'python3 ./classifier/DFM_KMeans.py ./feature/%s/%s.pickle %s %f 3 -outLogPickle %s/%s_log.pickle --preprocess -method minmax > %s/%s_result.csv' % (f, data, wvFile, nClusters, rDir, task, rDir, task)
            cmd = 'python3 ./classifier/DFM_KMeans.py ./feature/%s/%s.pickle %s %f 3 --preprocess -method minmax > %s/%s.csv' % (f, data, wvFile, nClusters, rDir, task)

            #print(cmd)
            cnt = cnt +1
            #sender.putTask(cmd)
            #if cnt >= 50:
            #    exit()
        
            cmd = 'python3 CollectResult.py %s/%s.csv >> %s' % (rDir, task, resultFile)
            #print(cmd)
            #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))

### run direct feature merging (using KMeans) for merged feature ###
nCEachTopic = {    
    3: { "Word": 0.60, "BiWord": 0.63, "TriWord": 0.60, "H/T_N": 0.73, "H/T_P": 0.73, "HO/OT": 0.44 },
    4: { "Word": 0.47, "BiWord": 0.32, "TriWord": 0.27, "H/T_N": 0.48, "H/T_P": 0.48, "HO/OT": 0.32 },
    5: { "Word": 0.63, "BiWord": 0.41, "TriWord": 0.29, "H/T_N": 0.61, "H/T_P": 0.61, "HO/OT": 0.36 },
    13: { "Word": 0.32, "BiWord": 0.52, "TriWord": 0.56, "H/T_N": 0.50, "H/T_P": 0.50, "HO/OT": 0.60 }
}

preprocess = 'minmax'
resultFile = 'Merge_DFM_KMeans_20160131.csv'
rDir2 = 'merge_DFM_result'
for t in [3, 4, 5, 13]:
    fFile = './feature/merge2_noPOS/t%d_merge2_noPOS_df2.pickle' % (t)
    for tDiff in range(-10, 11, 1):
        task = 't%d_merge_%s_Diff%g' % (t, preprocess, tDiff)
        #logFile = '%s/%s.pickle' % (rDir2, task)
        rFile = '%s/%s.csv' % (rDir2, task)
        cmd = 'python3 ./classifier/DFM_KMeans.py %s %s 1 3 --nCluster' % (fFile, wvFile)

        for fType, nC in nCEachTopic[t].items():
            cmd += ' %s %f' % (fType, nC + tDiff / 100.0 )
        cmd += ' --preprocess -method minmax > %s' % (rFile) 
        print(cmd)
        #sender.putTask(cmd)
            
        #cmd = 'python3 CollectResult.py %s/%s >> %s' % (libDir, rFile, resultFile)
        #print(cmd)
        #os.system(cmd)
    #os.system('echo "" >> %s' % (resultFile))



'''
### run feature merging (using community detection) for single feature ###
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




### run feature merging (community detection) for merged feature ###
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

'''
