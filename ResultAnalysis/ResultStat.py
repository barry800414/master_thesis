
import sys, re
import numpy as np
import matplotlib.pyplot as plt

# the module is to calculate basic statistical information from results
# including, 1. max  2. average  3. standard deviation  [4. plot the figure if needed]

# read CSV into a list of lists 
# each inner list represents a group of results under same topic 
def readCSV(filename):
    rowsList = list()
    rows = list()
    with open(filename, 'r') as f:
        for line in f:
            e = line.strip().split(',')
            if len(e) == 1 or len(e) == 0:
                if len(rows) != 0:
                    rowsList.append(rows)
                rows = list()
            else:
                rows.append((e[0], int(e[1]), float(e[2]), float(e[3]), float(e[4])))
                            # name,  dim,       train,        val,         test
    return rowsList
            
# filter out the rows with give string in the give column
def filterRows(rows, colIndex, keyword):
    newRows = list()
    for row in rows:
        if row[colIndex].find(keyword) == -1:
            newRows.append(row)
    return newRows

# extract threshold information from file name
def extractThreshold(string):
    r = re.search(".*_T(.+)_.*", string)
    if r is None:
        return None
    else:
        return float(r.group(1))

#FIXME: may have problems
def extractTaskName(string):
    p1 = string.rfind('/')
    s1 = string[p1+1:]
    s2 = re.sub("_T.+?_result.csv", '', s1)
    return s2

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], 'csv', file=sys.stderr)
        exit(-1)

    rowsList = readCSV(sys.argv[1])

    print('TaskName, BestThreshold, TestOfBestThreshold, TestScoreAvg, TestScoreStd')
    for rows in rowsList:
        # sort the rows based on validation scores
        sorted_rows = sorted(rows, key=lambda x:x[3], reverse=True) 
        n = extractTaskName(sorted_rows[0][0])
        t = extractThreshold(sorted_rows[0][0])
        best = sorted_rows[0][4]
        avg = np.mean([row[4] for row in sorted_rows])
        std = np.std([row[4] for row in sorted_rows])

        print(n, t, best, avg, std, sep=',')
        #print(name, rows[0][1], rows[0][2], rows[0][3], rows[0][4], sep=',')
    #print('')

