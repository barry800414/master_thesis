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
libFolder = '~/master_thesis/classifier/svm-sle'
seed = 1
for t in [3, 4, 5, 13]:
    for size in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        for fold in range(0, 10):
            cmd = 'cd %s; python3 runSVMSLE.py %d %d %d %d > ./result/T%dS%dF%dSize%d.csv' % (libFolder, t, fold, seed, size, t, seed, fold, size)
            print(cmd)
            sender.putTask(cmd)       
