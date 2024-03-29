#!/usr/bin/env python3
from multiprocessing.managers import BaseManager
import sys
import os

# declare the QueueManager like server
class QueueManager(BaseManager):
    pass
# register method get_queue()
QueueManager.register('get_queue')

if __name__ == '__main__':
    port = 3333
    if len(sys.argv) == 2:
        port = int(sys.argv[1])
    server_address = ('140.112.31.187', port)
    authkey = b'barry800414'
    m = QueueManager(address = server_address, authkey = authkey)
    print('Connecting to server (port:%d) ...' % (port), file=sys.stderr)
    m.connect()
    queue = m.get_queue()

    print(queue.qsize())
