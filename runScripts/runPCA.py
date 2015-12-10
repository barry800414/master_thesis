
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


fDir = '../feature'
libDir = '~/master_thesis/dimReduction'

resultFile = 'BOW_tf_PCA_20150712.csv'
for t in [3, 4, 5, 13]:
    for f in ['BOW_tf_shareVolc']: 
        for nComp in [None, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        #cmd = 'cd %s; python3 PCA.py %s %s/%s/t3_%s %s/%s/t4_%s %s/%s/t5_%s %s/%s/t13_%s' % (libDir, nComp, fDir, f, f, fDir, f, f, fDir, f, f, fDir, f, f)
        #print(cmd)
            cmd = 'python3 ./classifier/Run.py ./feature/%s/t%d_%s_PCA%s.pickle 3 > ./classifier/PCA/t%d_%s_PCA%s.csv' % (f, t, f, str(nComp), t, f, str(nComp))
            #print(cmd)
            #sender.putTask(cmd)
        
            cmd = 'python3 CollectResult.py ./classifier/PCA/t%d_%s_PCA%s.csv >> %s' % (t, f, str(nComp), resultFile)
            print(cmd)
            os.system(cmd)
    os.system('echo "" >> %s' % (resultFile))
