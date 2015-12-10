
from multiprocessing.managers import BaseManager
import os, sys

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

#sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)

# BOW_tf: BOW
# 2Word: Bi
# 3Word: Tri
# PT_SB: Dep_PP
# Full_SB: Dep_Full (Full = full representation)
# PT_TB: Dep_PPAll (all = TB tag based)
# Full_TB: Dep_FullAll 


resultFile = 'Single_DFC_Kmeans_20151210'
wvFile = './classifier/news7852Final.vector'
rDir = './classifier/single_DFC_KMeans'
for f in ['BOW_tf', '2Word', '3Word', 'Dep_PP', 'Dep_Full', 'Dep_PPAll', 'Dep_FullAll']: 
    for t in [3, 4, 5, 13]:
        for nClusters in [i * 0.01 for i in range(25, 76)]:
            data = 't%d_%s_df2' % (t, f)
            task = '%s_minmax' % (data)
            cmd = 'python3 ./classifier/DFC_KMeans.py ./feature/%s/%s.pickle %s %f 3 -outLogPickle %s/%s_log.pickle --preprocess -method minmax > %s/%s_result.csv' % (f, data, wvFile, nClusters, rDir, task, rDir, task)
            print(cmd)
            #sender.putTask(cmd)
        
            #cmd = 'python3 CollectResult.py %s/%s_result.csv >> %s' % (rDir, task, resultFile)
            #print(cmd)
            #os.system(cmd)
        #os.system('echo "" >> %s' % (resultFile))

