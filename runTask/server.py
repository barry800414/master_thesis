#!/usr/bin/env python3
from multiprocessing.managers import BaseManager
import queue
import sys

if __name__ == '__main__':
    port = 3333
    if len(sys.argv) == 2:
        port = int(sys.argv[1])

    q = queue.Queue()
    # a QueueManager hold a queue q, which automatically handle race condition
    class QueueManager(BaseManager):
        pass
    QueueManager.register('get_queue', callable = lambda: q)

    m = QueueManager(address = ('0.0.0.0', port), authkey = b'barry800414')
    s = m.get_server()
    print('Server is running now (port:%d) ...' % (port), file=sys.stderr)
    s.serve_forever()
