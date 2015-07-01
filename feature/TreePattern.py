
import json
import networkx as nx
import DepTree 

# The class for matching given patterns on dependency tree
class TreePattern():
    # first node is root 
    # word, tag, rel fields are originally list, here we convert them to set
    def __init__(self, name, nodeList, edgeList, rootId=0):
        self.p = nx.DiGraph()
        self.rootId = rootId
        self.name = name
        for i, n in enumerate(nodeList):
            if 'output_type' not in n:
                n['output_type'] = 'word'
            
            self.p.add_node(i, 
                    output_as=n['output_as'],
                    output_type=n['output_type'])
            assert 'tag' in n or 'deniedTag' in n
            assert 'word' in n or 'deniedWord' in n
            if 'tag' in n:
                self.p.node[i]['tag'] = TreePattern.initAllowedSet(n['tag'])
            if 'word' in n:
                self.p.node[i]['word'] = TreePattern.initAllowedSet(n['word'])
            if 'denialTag' in n:
                self.p.node[i]['deniedTag'] = TreePattern.initAllowedSet(n['deniedTag'])
            if 'denialWord' in n:
                self.p.node[i]['deniedWord'] = TreePattern.initAllowedSet(n['deniedWord'])
        
        for e in edgeList:
            self.p.add_edge(e['from'], e['to'], 
                    rel=TreePattern.initAllowedSet(e['rel']))

    def initAllowedSet(config):
        if config is None:
            # no configuration is specified, no constraint
            return None
        elif type(config) == list:
            # list of words or list of tags
            return set(config)
        elif type(config) == str:
            # filename to read the list of words or tags
            allowedSet = set()
            with open(config, 'r') as f:
                line = f.readline()
                entry = line.strip().split(',')
                for e in entry:
                    allowedSet.add(e.strip())
            return allowedSet

    # using this pattern to match the dependency trees
    def match(self, depTree, returnMapping=True):
        results = list()
        for tn in depTree.t.nodes():
            if TreePattern.matchNode(depTree.t.node[tn], 
                    self.p.node[self.rootId]):
                pEdgeStack = list(self.p.out_edges(self.rootId))
                if len(pEdgeStack) == 0:
                    continue
                pEdge = pEdgeStack.pop()
                tMatchedNodes = set([tn])
                mapping = { self.rootId : tn }
                TreePattern.matchTree(depTree, self, pEdge, 
                        tMatchedNodes, pEdgeStack, mapping, 
                        results, returnMapping)
        return results

    # recursively match the pattern on dependency tree, one call is for 
    # matching one edge
    # tTree: target dependency tree
    # pTree: pattern tree
    # pEdge: the edge to be matched in this step
    # tMatchedNodes: the set of matched nodes in target tree
    # pEdgeStack: the edges to be matched in the future
    # results: the list to store results
    # mapping: matching node pairs between pattern tree and dependency tree.
    #          it is a dict a -> b, where a is node id of pattern 
    #          tree and b is node id of target tree
    def matchTree(tTree, pTree, pEdge, tMatchedNodes, pEdgeStack, 
            mapping, results, returnMapping=True):
        # matching success
        (p1, p2) = pEdge
        t1 = mapping[p1] # find the mapped nodes in target tree
        for te in tTree.t.out_edges(t1):
            t2 = te[1]
            if t2 in tMatchedNodes:
                continue
            if TreePattern.matchEdge(tTree.t.edge[t1][t2], pTree.p.edge[p1][p2]):
                if TreePattern.matchNode(tTree.t.node[t2], pTree.p.node[p2]):
                    # prepare for next step (next state) 
                    tMatchedNodes.add(t2)
                    pEdgeStack.extend(pTree.p.out_edges(p2))
                    appendLen = len(pTree.p.out_edges(p2))
                    mapping[p2] = t2
                    
                    if len(pEdgeStack) != 0:
                        nextPEdge = pEdgeStack.pop()
                        TreePattern.matchTree(tTree, pTree, nextPEdge, 
                                tMatchedNodes, pEdgeStack, mapping, results)
                        # restore 
                        pEdgeStack.append(nextPEdge)
                    elif len(mapping) == pTree.p.number_of_nodes():
                        # match success
                        results.append(TreePattern.getMatchResult(tTree, pTree, 
                            mapping, returnMapping))
                    else:
                        if len(mapping) != pTree.p.number_of_nodes():
                            pass
                            #print('mapping:', mapping)
                            #print('#nodes:', pTree.p.number_of_nodes())

                    # restore to original state
                    tMatchedNodes.remove(t2)
                    pEdgeStack = pEdgeStack[:(-1)*appendLen]
                    del mapping[p2]


    # to check whether the node (in dependency tree) match the 
    # rules of the node (in pattern tree)
    def matchNode(tNode, pNode):
        if 'word' in pNode and pNode['word'] is not None:
            if tNode['word'] not in pNode['word']:
                return False
        elif 'denialWord' in pNode and pNode['denialWord'] is not None:
            if tNode['word'] in pNode['denialWord']:
                return False
            
        if 'tag' in pNode and pNode['tag'] is not None:
            if tNode['tag'] not in pNode['tag']:
                return False
        elif 'denialTag' in pNode and pNode['denialTag'] is not None:
            if tNode['tag'] in pNode['denialTag']:
                return False

        return True

    # to check whether the edge (in dependency tree) match the 
    # rules of the node (in pattern tree)
    def matchEdge(tEdge, pEdge):
        if pEdge['rel'] != None:
            if tEdge['rel'] not in pEdge['rel']:
                return False
        return True
    
    # get the matched result
    def getMatchResult(tTree, pTree, mapping, returnMapping=True):
        r = dict()
        for pId, tId in mapping.items():
            r[pTree.p.node[pId]['output_as']] = tTree.t.node[tId]['word']
            # only when output type is pos tagger, the tagger information will be captured
            if pTree.p.node[pId]['output_type'] in ['tag', 'Tag', 'pos','POS']:
                r[pTree.p.node[pId]['output_as'] + '_tag'] = tTree.t.node[tId]['tag']
        if returnMapping:
            r['mapping'] = dict(mapping)
        return r

def loadPatterns(filename):
    with open(filename, 'r') as f:
        patternList = json.load(f)
    pTreeList = list()
    for p in patternList:
        pTreeList.append(TreePattern(p['name'], p['nodes'], p['edges']))
    return pTreeList

# unit test
if __name__ == '__main__':
    # sample dependency tree
    tdList = ['nsubj 2 hate VBP 1 I PRP', 
              'root 0 ROOT null 2 hate VBP', 
              'det 4 product NN 3 this DT', 
              'dobj 2 hate VBP 4 product NN']

    depTree = DepTree.DepTree(tdList)
    
    # sample pattern tree
    pTreeDict = { 
            'nodes': [
                {'word':None, 'tag':None, 'output_as':'opinion'},
                {'word':None, 'tag':None, 'output_as':'holder'},
                {'word':None, 'tag':None, 'output_as':'target'}
            ],
            'edges': [
                {'from': 0, 'to': 1, 'rel': ['nsubj']}, 
                {'from': 0, 'to': 2, 'rel': ['dobj']}
            ],
            'name': 'holder-opinion-target'
    }

    pTree = TreePattern(pTreeDict['name'], pTreeDict['nodes'], pTreeDict['edges'])

    print(pTree.match(depTree))

    '''
    from NLPToolRequests import *
    sList = ['I hate this product',
             'Although I hate this product, Amy likes the product']
    for s in sList:
        tdList = sendDepParseRequest(s, seg=True, draw=True, fileFolder='.', fileName=s)
        #print(tdList)
        depTree = DepTree.DepTree(tdList)
        print(pTree.match(depTree))
    '''
