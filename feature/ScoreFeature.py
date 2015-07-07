
# This module implement supervised score features

# sum over the probability P(+|f) and P(-|f)
class PScore():
    def __init__(self, X, y):
        indexList = self.getClassIndexList(y)
        

    def transform(X):
        pass
    
    # get index list of each class
    def getClassIndexList(self, y):
        indexList = [list() for i in range(0, max(y)+1)]
        for i, yi in enumerate(y):
            indexList[yi].append(i)
        return indexList

