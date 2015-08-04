
import sys


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
            
# for string
def filterRows(rows, colIndex, keyword):
    newRows = list()
    for row in rows:
        if row[colIndex].find(keyword) == -1:
            newRows.append(row)
    return newRows

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], 'csv', file=sys.stderr)
        exit(-1)

    rowsList = readCSV(sys.argv[1])
    
    for rows in rowsList:
        rows = filterRows(rows, 0, 'uX1')
        if len(rows) == 0:
            continue
        rows.sort(key=lambda x:x[3], reverse=True) 
        p = rows[0][0].rfind('/')
        if p != -1:
            name = rows[0][0][p+1:]
        else:
            name = rows[0][0]

        print(name, rows[0][1], rows[0][2], rows[0][3], rows[0][4], sep=',')
    print('')
