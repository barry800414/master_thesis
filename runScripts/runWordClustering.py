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

sender = SendJob()

ans = input("Sure to run ? (Y/N)")
if ans != 'Y':
    print('exit', file=sys.stderr)
    exit(0)

libDir = '~/master_thesis/dimReduction/WordClustering'
AD_T = JJ_T = VA_T = 0.3
vFile = 'news7852Final.vector' 
sFile = '../../res/NTUSD_core.csv'
tFile = 'wordTag_news7852.txt'

for t in [3,4,5,13]:
    for VV_T in range(30, 41, 2):
        for NN_T in range(30, 41, 2):
            cmd = "cd %s; python3 WordMerge.py %s t%d.volc %s %s ./volc/t%d_VV%d_NN%d -VV %g -NN %g -AD %g -JJ %g -VA %g -NR 1.0" % (libDir, vFile, t, sFile, tFile, t, VV_T, NN_T, float(VV_T)/100, float(NN_T)/100, AD_T, JJ_T, VA_T)
            print(cmd)
            sender.putTask(cmd)

