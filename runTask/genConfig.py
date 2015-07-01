

import copy
import json

configFolder = './config/'
topicList = [2, 3, 4, 5, 13]

####### generating basic volc configs #########
volcFolder = './fSelect'
basicVolcConfig = { "WM": dict(), "Dep_Full": dict(), "Dep_POS": dict(), 
        "Dep_PP": dict(), "Dep_PPAll": dict(), "OM_noH":dict(), 
        "OM_withH": dict(), "OM_stance": dict() }
k2 = 40
k1 = 'tfidf'
# for WM
c = basicVolcConfig['WM']
for t in topicList:
    c[t] = dict()
    c[t]['main'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)
# for OM
for model in ['Dep_Full', 'Dep_POS', 'Dep_PP', 'Dep_PPAll', 'OM_noH', 'OM_withH', 'OM_stance']:
    c = basicVolcConfig[model]
    for t in topicList:
        c[t] = dict()
        c[t]['holder'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)
        c[t]['opinion'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)
        c[t]['target'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)

# default config of each model
targetScore = "Accuracy"
nfolds = 20
testSize = 0.1
defaultConfig={
        "WM": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "feature": ["tfidf"],
                "allowedPOS": [["VA", "VV", "NN", "NR", "AD", "JJ"]]
            },
            "setting":{
                "targetScore": targetScore,
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "volc": basicVolcConfig['WM'],
            "wordGraph": None
        },
        "Dep_Full": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "keyTypeList": [["OT"]],
                "opnNameList": [None],
                "negSepList": [[True]],
                "ignoreNeutral": [False],
                "pTreeSepList": [[False]],
                "countTreeMatched": [False]
            },
            "setting":{
                "targetScore": targetScore,
                "targetScore": "Accuracy",
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "treePattern": "./DepBased/pattern_Dep.json",
            "volc":  basicVolcConfig['Dep_Full'],
            "phrase": None,
            "wordGraph": None
        },
        "Dep_POS": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "keyTypeList": [["OT"]],
                "opnNameList": [None],
                "negSepList": [[True]],
                "ignoreNeutral": [False],
                "pTreeSepList": [[False]],
                "countTreeMatched": [False]
            },
            "setting":{
                "targetScore": targetScore,
                "targetScore": "Accuracy",
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "treePattern": "./DepBased/pattern_DepPOS.json",
            "volc":  basicVolcConfig['Dep_POS'],
            "phrase": None,
            "wordGraph": None
        },
        "Dep_PP": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "keyTypeList": [["T"]],
                "opnNameList": [None],
                "negSepList": [[True]],
                "ignoreNeutral": [False],
                "pTreeSepList": [[False]],
                "countTreeMatched": [False]
            },
            "setting":{
                "targetScore": targetScore,
                "targetScore": "Accuracy",
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "treePattern": "./DepBased/pattern_Dep.json",
            "volc":  basicVolcConfig['Dep_PP'],
            "phrase": None,
            "wordGraph": None
        },
        "Dep_PPAll": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "keyTypeList": [["T"]],
                "opnNameList": [None],
                "negSepList": [[True]],
                "ignoreNeutral": [False],
                "pTreeSepList": [[False]],
                "countTreeMatched": [False]
            },
            "setting":{
                "targetScore": targetScore,
                "targetScore": "Accuracy",
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "treePattern": "./DepBased/pattern_DepAll.json",
            "volc":  basicVolcConfig['Dep_PPAll'],
            "phrase": None,
            "wordGraph": None
        },
        "OM_noH": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "keyTypeList": [["T", "OT"]],
                "opnNameList": [None],
                "negSepList": [[True]],
                "ignoreNeutral": [False],
                "pTreeSepList": [[False]],
                "countTreeMatched": [True]
            },
            "setting":{
                "targetScore": targetScore, 
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "treePattern": "./DepBased/pattern_noH.json",
            "volc":  basicVolcConfig['OM_noH'],
            "phrase": None,
            "wordGraph": None
        },
        "OM_withH": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "keyTypeList": [["H", "T", "HT", "HO", "OT"]],
                "opnNameList": [None],
                "negSepList": [[True]],
                "ignoreNeutral": [False],
                "pTreeSepList": [[False]],
                "countTreeMatched": [True]
            },
            "setting":{
                "targetScore": targetScore, 
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "treePattern": "./DepBased/pattern_withH.json",
            "volc": basicVolcConfig['OM_withH'],
            "phrase": None,
            "wordGraph": None
        },
        "OM_stance": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "keyTypeList": [["H", "T", "HT", "HO", "OT"]],
                "opnNameList": [None],
                "negSepList": [[True]],
                "ignoreNeutral": [False],
                "pTreeSepList": [[False]],
                "countTreeMatched": [True]
            },
            "setting":{
                "targetScore": targetScore, 
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "treePattern": "./DepBased/pattern_stance.json",
            "volc": basicVolcConfig['OM_stance'],
            "phrase": None,
            "wordGraph": None
        },
        "WM_LDA": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
                "nTopics": [100],
                "nIters": [300],
                "feature": ["tf"]
            },
            "setting":{
                "targetScore": targetScore, 
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            #"volc": basicVolcConfig['WM'],
            "volc": None
        },
        "Merged": {
            "toRun": ["SelfTrainTest"],
            #"toRun": ["SelfTrainTest", "AllTrainTest", "LeaveOneTest"],
            #"preprocess": { "method": "binary", "params": { "threshold": 0.0 }},
            "preprocess": { "method": "minmax", "params": { "feature_range": [0,1] }},
            "minCnt": 4,
            "params":{ 
            },
            "setting":{
                "targetScore": targetScore,
                "clfName": "MaxEnt",
                "randSeedList": [i for i in range(1,31)],
                "testSize": testSize,
                "n_folds": nfolds,
                "fSelectConfig": None
            },
            "volc": None,
            "wordGraph": None
        }

}

####### generate configs for word graph ########
wgFolder = './WordClustering/wordGraph'
k1 = 20
beta = 0.75
step = 2

wgConfigEachModel = dict()

for model in ['WM', 'OLDM_Full', 'other']:
    wgConfig = dict()
    if model == 'WM':
        k2Range = [2, 3, 5, 10]
    else:
        k2Range = [2, 3, 5]
    for method in ['TopKEachRow', 'TopK']:
        for k2 in k2Range:
            name = 'wg7852_top%d_beta%d_step%d_%s%d' % (k1, round(beta*100), step, method, k2)
            wgConfig[name] = dict()
            for t in topicList:
                wgVolcConfig = { 
                    "WM": {
                        "main": "%s/news7852Final_T%d_P40.volc" % (wgFolder, t) 
                    },
                    "OLDM_Full": {
                        "seed": "%s/news7852Final_T%d_P40.volc" % (wgFolder, t), 
                        "firstLayer": "%s/news7852Final_T%d_P40.volc" % (wgFolder, t) 
                    },
                    "other": {
                        "holder":  "%s/news7852Final_T%d_P40.volc" % (wgFolder, t),
                        "target": "%s/news7852Final_T%d_P40.volc" % (wgFolder, t),
                        "opinion": "%s/news7852Final_T%d_P40.volc" % (wgFolder, t) 
                    }
                }
                wgConfig[name][t] = { 
                    "filename": "%s/%s.mtx" % (wgFolder, name),
                    "params": {},
                    "volcFile": wgVolcConfig[model]
                }
    name = 'wg7852_top%d_None' % (k1)
    wgConfig[name] = dict()
    for t in topicList:
        wgConfig[name][t] = { 
            "filename": "%s/%s.mtx" % (wgFolder, name),
            "params": {},
            "volcFile": wgVolcConfig[model]
    }
    wgConfigEachModel[model] = wgConfig

####### generating configs for word clustering  #########
volcFolder = './fSelect'
volcFileConfig = { "WM": dict(), "OLDM_Full": dict(), "OLDM_PP": dict(), "OM_noH":dict(), "OM_withH": dict() }
# for WM
c = volcFileConfig['WM']
for k1 in ['tf', 'df', 'tfidf']:
    for k2 in range(10, 101, 10):
        type = '%s_P%03d' % (k1, k2)
        c[type] = dict()
        for t in topicList:
            c[type][t] = dict()
            c[type][t]['main'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)
# for OLDM
c = volcFileConfig['OLDM_Full']
for k1 in ['tf', 'df', 'tfidf']:
    for k2 in range(10, 101, 10):
        type = '%s_P%03d' % (k1, k2)
        c[type] = dict()
        for t in topicList:
            c[type][t] = dict()
            c[type][t]['seed'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)
            c[type][t]['firstLayer'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)
# for OM
for model in ['OLDM_PP', 'OM_noH', 'OM_withH']:
    c = volcFileConfig[model]
    for k1 in ['tf', 'df', 'tfidf']:
        for k2 in range(10, 101, 10):
            type = '%s_P%03d' % (k1, k2)
            c[type] = dict()
            for t in topicList:
                c[type][t] = dict()
                c[type][t]['holder'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)
                c[type][t]['opinion'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)
                c[type][t]['target'] = '%s/%s_T%d_P%d.volc' % (volcFolder, k1, t, k2)

# LDA:
nTopicConfig = dict()
for nTopics in [20, 40, 60, 80, 100]:
#for nTopics in [20, 30, 40, 50, 75, 100, 200, 300, 500]:
    nTopicConfig["nT%03d" % (nTopics)] = [nTopics]
nIterConfig = dict()
for nIters in [100, 300, 500]:
    nIterConfig["nI%03d" % (nIters)] = [nIters]

# configuration for search parameters (one parameter a time)
iterConfig={
    "WM": [
        #{ "path": ["preprocess"], 
        #    "params": { #"None": None,
                       #"binary": { "method": "binary", "params": { "threshold": 0.0 }},
        #               "minmax": { "method": "minmax", "params": { "feature_range": [0,1] }}
        #               }
        #    },
        #{ "path": ["wordGraph"],
        #    "params": wgConfigEachModel['WM']
        #}
        { "path": ["minCnt"], 
            "params": { "m5": 6, "m7": 8, "m10": 11 }
        }
    ],
    "Dep_Full" :[
        #{ "path": ["preprocess"], 
        #    "params": {#"None": None,
        #              #"binary": { "method": "binary", "params": { "threshold": 0.0 }},
        #               "minmax": { "method": "minmax", "params": { "feature_range": [0,1] }}
        #               }
        #    },
        #{ "path": ["wordGraph"],
        #    "params": wgConfigEachModel['OLDM_Full']
        #}
        { "path": ["minCnt"], 
            "params": { "m5": 6, "m7": 8, "m10": 11 }
        }

    ],
    "Dep_POS":[  
        #{ "path": ["preprocess"], 
        #    "params": { #"None": None,
                        #"binary": { "method": "binary", "params": { "threshold": 0.0 }},
        #                "minmax": { "method": "minmax", "params": { "feature_range": [0,1] }}
        #               }
        #    },
        #{ "path": ["wordGraph"],
        #    "params": wgConfigEachModel['other']
        #}
        { "path": ["minCnt"], 
            "params": { "m5": 6, "m7": 8, "m10": 11 }
        }

    ],
    "Dep_PP":[  
        #{ "path": ["preprocess"], 
        #    "params": { #"None": None,
                        #"binary": { "method": "binary", "params": { "threshold": 0.0 }},
        #                "minmax": { "method": "minmax", "params": { "feature_range": [0,1] }}
        #               }
        #    },
        #{ "path": ["wordGraph"],
        #    "params": wgConfigEachModel['other']
        #}
        { "path": ["minCnt"], 
            "params": { "m5": 6, "m7": 8, "m10": 11 }
        }
    ],
    "Dep_PPAll":[  
        #{ "path": ["preprocess"], 
        #    "params": { #"None": None,
                        #"binary": { "method": "binary", "params": { "threshold": 0.0 }},
        #                "minmax": { "method": "minmax", "params": { "feature_range": [0,1] }}
        #               }
        #    },
        #{ "path": ["wordGraph"],
        #    "params": wgConfigEachModel['other']
        #}
        { "path": ["minCnt"], 
            "params": { "m5": 6, "m7": 8, "m10": 11 }
        }
    ],
    "OM_noH":[  
        #{ "path": ["preprocess"], 
        #    "params": { #"None": None,
                        #"binary": { "method": "binary", "params": { "threshold": 0.0 }},
        #                "minmax": { "method": "minmax", "params": { "feature_range": [0,1] }}
        #               }
        #    },
        #{ "path": ["params", "keyTypeList"], 
        #    "params": { "T": [["T"]], "OT": [["OT"]] },
        #    },
        #{ "path": ["params", "pTreeSepList"],
        #    "params": { "pTreeBoth": [[False, True]], "pTreeSep": [[True]] }
        #},
        #{ "path": ["params", "countTreeMatched"],
        #    "params": { "noCnt": [False] } 
        #},
        #{ "path": ["wordGraph"],
        #    "params": wgConfigEachModel['other']
        #}
        { "path": ["minCnt"], 
            "params": { "m5": 6, "m7": 8, "m10": 11 }
        }
    ],
    "OM_withH":[  
        #{ "path": ["preprocess"], 
        #    "params": { #"None": None,
                        #"binary": { "method": "binary", "params": { "threshold": 0.0 }},
        #                "minmax": { "method": "minmax", "params": { "feature_range": [0,1] }}
        #               }
        #    },
        #{ "path": ["params", "keyTypeList"], 
        #    "params": { "T": [["T"]], "H": [["H"]], "HT":[["HT"]], "HOT": [["HOT"]], 
        #        "OT": [["OT"]], "HO":[["HO"]], "all": [["H", "T", "OT", "HO", "HOT", "HT"]],
        #        "Tall": [["OT", "T"]], "Hall": [["HO", "H"]], "HTall": [["HO","H","T","OT"]] },
        #    },
        { "path": ["params", "keyTypeList"], 
            "params": { "HTall": [["HO","H","T","OT"]] },
            },
        #{ "path": ["params", "pTreeSepList"],
        #    "params": { "pTreeBoth": [[False, True]], "pTreeSep": [[True]] }
        #},
        #{ "path": ["params", "countTreeMatched"],
        #    "params": { "noCnt": [False] } 
        #},
        #{ "path": ["wordGraph"],
        #    "params": wgConfigEachModel['other']
        #}
        { "path": ["minCnt"], 
            "params": { "m5": 6, "m7": 8, "m10": 11 }
        }
    ],
    "OM_stance":[  
        #{ "path": ["preprocess"], 
        #    "params": { #"None": None,
                        #"binary": { "method": "binary", "params": { "threshold": 0.0 }},
        #                "minmax": { "method": "minmax", "params": { "feature_range": [0,1] }}
        #               }
        #    },
        #{ "path": ["params", "keyTypeList"], 
        #    "params": { "T": [["T"]], "H": [["H"]], "HT":[["HT"]], "HOT": [["HOT"]], 
        #        "OT": [["OT"]], "HO":[["HO"]], "all": [["H", "T", "OT", "HO", "HOT", "HT"]],
        #        "Tall": [["OT", "T"]], "Hall": [["HO", "H"]], "HTall": [["HO", "H", "T", "OT"]] },
        #    },
        { "path": ["params", "keyTypeList"], 
            "params": { "HTall": [["HO","H","T","OT"]] },
            },
        #{ "path": ["params", "pTreeSepList"],
        #    "params": { "pTreeBoth": [[False, True]], "pTreeSep": [[True]] }
        #},
        #{ "path": ["params", "countTreeMatched"],
        #    "params": { "noCnt": [False] } 
        #},
        #{ "path": ["wordGraph"],
        #    "params": wgConfigEachModel['other']
        #}
        { "path": ["minCnt"], 
            "params": { "m5": 6, "m7": 8, "m10": 11 }
        }
    ],

    "WM_LDA":[  
        { "path": ["params", "nTopics"],
            "params": nTopicConfig
        },
        { "path": ["params", "nIters"],
            "params": nIterConfig
        }
    ]

}

nameList= {
    #"WM": [ "mm"], #pre, clf
    #"Dep_Full":  ["mm" ], #pre, clf
    #"Dep_POS": [ "mm"],
    #"Dep_PP": [ "mm" ], #pre, clf
    #"Dep_PPAll": [ "mm" ], #pre, clf
    #"OM_noH": [ "mm", "pTreeBoth", "cnt" ], #pre, clf
    #"OM_withH": ["mm", "H-T-HT", "pTreeBoth", "cnt" ], # pre, clf, keyType,
    #"OM_stance": ["mm", "H-T-HT", "pTreeBoth", "cnt" ], # pre, clf, keyType,
    #"WM_LDA": ["nT100", "nI300"]

    #"OM_noH": [ "mm", "MaxEnt", "pTreeBoth" ], #pre, clf
    #"OM_withH": ["mm", "MaxEnt", "H-T-HT", "pTreeBoth" ] # pre, clf, keyType,
    #"WM": [ "binary", "N"], #pre, wg
    #"OLDM_Full":  ["binary", "Tag", "N"], #pre, feature, wg
    #"OLDM_PP": [ "binary", "igFalse", "N" ], #pre, ignore, wg
    #"OM_noH": [ "binary", "Tall", "igFalse", "N"], #pre, keyType, ignore, wg
    #"OM_withH": [ "binary", "H-T-HT", "igFalse", "N"] # pre, keyType, ignore, wg

    "WM": [ "m3" ], #pre, clf
    "Dep_Full":  ["m3" ], #pre, clf
    "Dep_POS": [ "m3"],
    "Dep_PP": [ "m3" ], #pre, clf
    "Dep_PPAll": [ "m3" ], #pre, clf
    "OM_noH": [ "m3" ], #pre, clf
    "OM_withH": [ "HTall", "m3" ], # pre, clf, keyType,
    "OM_stance": [ "HTall", "m3" ] # pre, clf, keyType,
}


def genConfig(defaultConfig, iterConfig, nameList, prefix):
    configList = list()
    for i in range(0, len(iterConfig)):
        path = iterConfig[i]["path"]
        params = iterConfig[i]["params"]
        for pName, p in sorted(params.items()):
            newConfig = copy.deepcopy(defaultConfig)
            newNameList = copy.deepcopy(nameList)
            
            # replace new configs
            obj = newConfig
            for j in range(0, len(path) - 1):
                obj = obj[path[j]]
            obj[path[-1]] = p
            newNameList[i] = pName
            newName = mergeName(prefix, newNameList)
            newConfig['taskName'] = newName
            configList.append( (newName, newConfig) )
    return configList


def genSingleConfig(defaultConfig, iterConfig, nameList, prefix, modelName, selectedList):
    assert len(nameList) == len(selectedList)
    newConfig = copy.deepcopy(defaultConfig[modelName])
    newNameList = copy.deepcopy(nameList)

    for i in range(0, len(selectedList)):
        pName = selectedList[i] 
        if pName is None:
            # keep default config
            continue
        path = iterConfig[i]['path']
        paramsDict = iterConfig[i]['params']
        if pName in paramsDict:
            obj = newConfig
            for j in range(0, len(path) - 1):
                obj = obj[path[j]]
            obj[path[-1]] = paramsDict[name]
            newNameList[i] = pName
        else:
            print(name, ' not found !', file=sys.stderr)

    newName = mergeName(prefix, newNameList)
    newConfig['taskName'] = newName
    return (newName, newConfig)

def mergeName(prefix, nameList):
    outStr = str(prefix)
    for n in nameList:
        outStr += '_' + n
    return outStr

def genMergedConfig(modelNameList):
    config = copy.deepcopy(defaultConfig['Merged'])
    taskName = ''
    for i, model in enumerate(modelNameList):
        if len(taskName) == 0:
            taskName = model
        else:
            taskName += '_' + model
    config['taskName'] = taskName
    return (taskName, config)

if __name__ == '__main__':
    suffix = '_TwoClass'
    suffix = '_5T_Merged'
    #suffix = '_7T_Merged_withNoLabel'
    #suffix = '_4T'
    #suffix = '_Filtered_5T_Merged'

    # for single model
    for model in ["WM", "Dep_Full", "Dep_POS", "Dep_PP", "Dep_PPAll", "OM_noH", "OM_withH", "OM_stance"]:
    #for model in ["WM_LDA"]:
        configList = genConfig(defaultConfig[model], iterConfig[model], nameList[model], prefix = model + suffix)
        print(model, len(configList))
        for name, config in configList:
            with open(configFolder + name + '_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            print(name)
            #print(config, '\n')
            pass

    mList = [
            ['WM', 'Dep_Full', 'Dep_POS', 'Dep_PP'],
            ['WM', 'Dep_Full', 'Dep_POS', 'Dep_PPAll'],
            ['WM', 'OM_noH'],
            ['WM', 'OM_withH'],
            ['WM', 'OM_stance'],
            ['WM', 'Dep_PPAll', 'OM_noH'],
            ['WM', 'Dep_PPAll', 'OM_withH'],
            ['WM', 'Dep_PPAll', 'OM_stance']
        ]

    for m in mList:
        name, config = genMergedConfig(m)
        #with open(configFolder + name + '_config.json', 'w') as f:
        #   json.dump(config, f, indent=2)
        #print(name)
        #print(config, '\n')


    '''
    suffix = '_4T'
    # for merged model
    # WM+OLDM, WM+OM, WM is fixed
    for model in ["OLDM_PP", "OM_withH"]:
        configList = genConfig(defaultConfig[model], iterConfig[model], nameList[model], prefix = 'WM_' + model + suffix)
        print("WM_"+ model, len(configList))
        for name, config in configList:
            #with open(configFolder + name + '_config.json', 'w') as f:
            #    json.dump(config, f, indent=2)
            #print(name)
            #print(config)
            pass

    # WM and OLDM is fixed
    for model in ["OM_withH"]:
        configList = genConfig(defaultConfig[model], iterConfig[model], nameList[model], prefix = 'WM_OLDM_PP_' + model + suffix)
        print("WM_OLDM_"+ model, len(configList))
        for name, config in configList:
            #with open(configFolder + name + '_config.json', 'w') as f:
            #    json.dump(config, f, indent=2)
            #print(name)
            #print(config)
            pass

    '''
