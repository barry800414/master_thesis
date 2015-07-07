
from multiprocessing.managers import BaseManager
import os

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

for t in [2, 3, 4, 5, 13]:
    resultFile = 'Single20150707_basic_t%d.csv' % (t)
    #for f in ['docLen', 'wordDiv', 'nUniqueWord', 'groupCnt', 'media']:
    for f in ['groupCnt_media', 'basicAll']:
        cmd = 'python3 ./classifier/Run.py ./feature/basic/t%d_%s.pickle 3 > ./classifier/basic/t%d_%s_results.csv' % (t, f, t, f)
        #print(cmd)
        #sender.putTask(cmd)
        #os.system(cmd)
    
        cmd = 'python3 CollectResult.py ./classifier/basic/t%d_%s_results.csv >> %s' % (t, f, resultFile)
        print(cmd)
        os.system(cmd)



