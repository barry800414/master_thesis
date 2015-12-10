import os

for i in range(0, 100):
    p = i*0.01
    cmd = 'python3 dsadasd.py %d.txt >> %d.txt' % (i, i)
    os.system(cmd)

