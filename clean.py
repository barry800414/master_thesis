
import sys

lines = list()
with open(sys.argv[1], 'r') as f:
    for i, line in enumerate(f):
        if i not in [11, 22]:
            lines.append(line.strip())

with open(sys.argv[1], 'w') as f:
    for line in lines:
        print(line, file=f)
