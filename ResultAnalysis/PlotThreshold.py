
import sys, re
import numpy as np
import matplotlib.pyplot as plt

# the module is to regorganize the data for excel, plot the figures if needed

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
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], 'csv [FiguresOutputFolder]', file=sys.stderr)
        exit(-1)

    rowsList = readCSV(sys.argv[1])
    
    # reorganize the data, #FIXME:assume all data has same threshold ranges
    print('TaskName', end='')
    for row in rowsList[0]:
        t = extractThreshold(row[0])
        print(', %f' % t, end='')
    print('')
    for rows in rowsList:
        taskName = extractTaskName(rows[0][0])
        print(taskName, end='')
        for row in rows:
            print(', %f' % row[4], end='')
        print('')

    if len(sys.argv) == 3:
        folder = sys.argv[2]
        for rows in rowsList:
            tList = [extractThreshold(row[0]) for row in rows]
            testList = [row[4] for row in rows]
            plt.plot(tList, testList)
            taskName = extractTaskName(rows[0][0])
            plt.title(taskName)
            plt.ylabel('Testing scores')
            plt.xlabel('Threshold of similarity to build edges')
            limit = plt.axis()
            plt.axis([limit[0], limit[1], 0.5, 0.9])  # x-axis: auto, y-axis: [0.5, 0.9)
            plt.savefig(folder + '/' + taskName + '.svg')
            plt.clf()

