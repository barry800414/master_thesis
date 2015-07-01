#!/usr/bin/env python3
from multiprocessing.managers import BaseManager
import os
from genConfig import *

class QueueManager(BaseManager):
    pass
class SendJob:
    def __init__(self):
        QueueManager.register('get_queue')
        port = 3333
        self.m = QueueManager(address=('140.112.187.33', 3333), authkey=b'barry800414')
        self.m.connect()
        self.queue = self.m.get_queue()

    def putTask(self, cmd):
        self.queue.put(cmd)

pyMap = { 
        "WM": "./baseline/WordModel_New.py", 
        "WM_LDA": "./fSelect/LDA_test.py",
        "Dep_Full": "./DepBased/OpinionModel_New.py", 
        "Dep_POS": "./DepBased/OpinionModel_New.py", 
        "Dep_PP": "./DepBased/OpinionModel_New.py",
        "Dep_PPAll": "./DepBased/OpinionModel_New.py",
        "OM_noH": "./DepBased/OpinionModel_New.py",
        "OM_withH": "./DepBased/OpinionModel_New.py",
        "OM_stance": "./DepBased/OpinionModel_New.py",
        "merged": "./DepBased/MergedModel.py",
}
configFolder = './config/'

#suffix = '_TwoClass'
#taggedFile = './zhtNewsData/taggedLabelNewsTwoClass_long.json'
#depFile = './zhtNewsData/DepParsedLabelNewsTwoClass_short.json'
#labelNewsFile = './zhtNewsData/taggedLabelNewsTwoClass_long.json' #FIXME

#suffix = '_Filtered_5T_Merged'
#suffix = '_Filtered_4T'
#taggedFile = './zhtNewsData/taggedLabelNews%s_long.json' % (suffix[1:])
#depFile = './zhtNewsData/DepParsedLabelNews%s_short.json' % (suffix[1:])
#labelNewsFile = taggedFile

#suffix = ''
#taggedFile = './zhtNewsData/taggedLabelNewsFinal_long.json'
#depFile = './zhtNewsData/DepParsedLabelNewsFinal_short.json'
#labelNewsFile = taggedFile

#suffix = '_4T'
suffix = '_5T_Merged'
#suffix = '_7T_Merged_withNoLabel'
taggedFile = './zhtNewsData/taggedLabelNews%s_long.json' %(suffix)
#depFile = './zhtNewsData/DepParsedLabelNews%s_short.json' %(suffix)
depFile = './zhtNewsData/DepParsedLabelNews%s_long.json' %(suffix)
labelNewsFile = taggedFile

#taggedDepFile = './zhtNewsData/taggedAndDepParsedLabelNews20150504.json'
dictFile = './res/NTUSD_core.csv'
negFile = './DepBased/negPattern.json'
#patternFile = './DepBased/my_pattern.json'

if __name__ == '__main__':
    sender = SendJob()
    
    # for single model
    #for model in ["WM", "Dep_Full", "Dep_POS", "Dep_PP", "OM_noH", "OM_withH"]:
    for model in ["WM", "Dep_Full", "Dep_POS", "Dep_PP", "Dep_PPAll", "OM_noH", "OM_withH", "OM_stance"]:
    #for model in ["WM_LDA"]:
        configList = genConfig(defaultConfig[model], iterConfig[model], nameList[model], prefix = model + suffix)
        print(model, len(configList))
        for taskName, config in configList:
            configFile = configFolder + taskName + '_config.json'
            resultFile = '%s_results.csv' % (taskName)
            if model in ['WM', 'WM_LDA']:
                cmd = "python3 %s %s %s > %s" % (pyMap[model], taggedFile, configFile, resultFile)
            elif model in ['Dep_PP', "Dep_Full", "Dep_POS", 'OM_noH', 'OM_withH', "OM_stance", "Dep_PPAll"]:
                cmd = "python3 %s %s %s %s %s > %s" % (pyMap[model], depFile, configFile, negFile, dictFile, resultFile)
            else:
                assert 1 != 1
            print(cmd)
            sender.putTask(cmd)
    
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
    pList = [
            ['WM_5T_Merged_minmax', 'Dep_Full_5T_Merged_minmax','Dep_POS_5T_Merged_minmax', 'Dep_PP_5T_Merged_minmax'],
            ['WM_5T_Merged_minmax', 'Dep_Full_5T_Merged_minmax','Dep_POS_5T_Merged_minmax', 'Dep_PPAll_5T_Merged_minmax'],
            ['WM_5T_Merged_minmax', 'OM_noH_5T_Merged_mm_pTreeNotSep_cnt'],
            ['WM_5T_Merged_minmax', 'OM_withH_5T_Merged_mm_Tall_pTreeBoth_cnt'],
            ['WM_5T_Merged_minmax', 'OM_stance_5T_Merged_mm_H-T-HT_pTreeNotSep_cnt'],
            ['WM_5T_Merged_minmax', 'Dep_PPAll_5T_Merged_minmax', 'OM_noH_5T_Merged_mm_pTreeNotSep_cnt'],
            ['WM_5T_Merged_minmax', 'Dep_PPAll_5T_Merged_minmax', 'OM_withH_5T_Merged_mm_Tall_pTreeBoth_cnt'],
            ['WM_5T_Merged_minmax', 'Dep_PPAll_5T_Merged_minmax', 'OM_stance_5T_Merged_mm_H-T-HT_pTreeNotSep_cnt']
        ]


    for i, m in enumerate(mList):
        taskName, config = genMergedConfig(m)
        configFile = configFolder + taskName + '_config.json'
        resultFile = '%s_results.csv' % (taskName)
        cmd = "python3 %s %s %s" % (pyMap['merged'], labelNewsFile, configFile)
        for j, modelName in enumerate(m):
            cmd += ' -%s %s' % (modelName, pList[i][j])
        cmd += ' > %s' % (resultFile)
        #print(cmd)
        #sender.putTask(cmd)


    '''
    suffix2 = '_4T'
    # for mixed model 
    #WMPickleFile = 'WM_TwoClass_pN_fN_MaxEnt_01_vN'
    #WMPickleFile = 'WM_binary_MaxEnt_tfidf_N_Top20-beta75-step10-select20'
    #WMPickleFile = 'WM_Filtered_5T_Merged_binary_MaxEnt_tfidf_N_N'
    #WMPickleFile = 'WM_4T_binary_MaxEnt_tfidf_N_N'
    #WMPickleFile = 'WM_4T_binary_MaxEnt_tfidf_N_Top20-beta75-step10-select20'
    WMPickleFile = 'WM_5T_Merged_binary_MaxEnt_tfidf_N_N'
    #WMPickleFile = 'WM_5T_Merged_binary_MaxEnt_tfidf_N_Top20-beta75-step10-select10'
    WMPickleFile = 'WM_4T_binary_N'
    for model in ["OLDM_PP", "OM_withH"]:
        configList = genConfig(defaultConfig[model], iterConfig[model], nameList[model], prefix = "WM_" + model + suffix2)
        print(model, len(configList))
        for taskName, config in configList:
            configFile = configFolder + taskName + '_config.json'
            resultFile = '%s_results.csv' % (taskName)
            cmd = "python3 %s %s %s -WM %s -%s %s > %s" % (pyMap['merged'], labelNewsFile, configFile, 
                    WMPickleFile, model, model + suffix + taskName[len("WM_"+ model +suffix2):], resultFile)
            #print(cmd)
            #sender.putTask(cmd)
   
    # for mixed model 
    #OLDMPickleFile = 'OLDM_TwoClass_pN_fN_MaxEnt_tag_vN'
    #OLDMPickleFile = 'OLDM_binary_MaxEnt_tag_N' 
    #OLDMPickleFile = 'OLDM_Filtered_5T_Merged_binary_MaxEnt_Tag_N'
    #OLDMPickleFile = 'OM_4T_binary_MaxEnt_H-T-HT_N_N_OLDM'
    OLDMPickleFile = 'OM_5T_Merged_binary_MaxEnt_H-T-HT_N_N_OLDM'
    OLDMPickleFile = 'OLDM_PP_4T_binary_igFalse_N'
    for model in ["OM_withH"]:
        configList = genConfig(defaultConfig[model], iterConfig[model], nameList[model], prefix = "WM_OLDM_PP_" + model + suffix2)
        print(model, len(configList))
        for taskName, config in configList:
            configFile = configFolder + taskName + '_config.json'
            resultFile = '%s_results.csv' % (taskName)
            cmd = "python3 %s %s %s -WM %s -OLDM %s -%s %s > %s" % (pyMap['merged'], labelNewsFile, configFile, 
                    WMPickleFile, OLDMPickleFile, model, model + suffix + taskName[len("WM_OLDM_PP_" + model+ suffix2):], resultFile)

            #print(cmd)
            #sender.putTask(cmd)
   '''

if __name__ == '__main2__':
    for model in ['WM', 'OLDM', 'OM', 'WM_OLDM', 'WM_OM', 'WM_OLDM_OM']:
    #for model in ['WM', 'OLDM', 'OM']:
    #for model in ['WM_OLDM', 'WM_OM', 'WM_OLDM_OM']:
        for pre in [None, 'std', 'norm1', 'norm2', 'binary']:
        #for pre in ['binary']:
            prefix = '%s_%s_None' % (model, pre)
            configFile = configFolder + '%s_config.json' % (prefix)
            resultFile = '%s_results.csv' % (prefix)
            WMParamFile = 'WM_%s_None_params.json' % (pre)
            OLDMParamFile = 'OLDM_%s_None_params.json' % (pre)
            OMParamFile = 'OM_%s_None_params.json' % (pre)
            if model == 'WM':
                cmd = "python3 %s %s %s > %s" % (pyMap[model], taggedFile, configFile, resultFile)
            elif model == 'OLDM':
                cmd = "python3 %s %s %s %s > %s" %(pyMap[model], depFile, configFile, dictFile, resultFile) 
            elif model == 'OM':
                cmd = "python3 %s %s %s %s %s %s > %s" % (pyMap[model], depFile, configFile, patternFile, negFile, dictFile, resultFile)
            elif model == 'WM_OLDM':
                cmd = "python3 %s %s %s %s -WM %s -OLDM %s > %s" % (pyMap[model], taggedDepFile, configFile, dictFile, WMParamFile, OLDMParamFile, resultFile)
            elif model == 'WM_OM':
                cmd = "python3 %s %s %s %s -WM %s -OM %s -tp %s -ng %s > %s" % (pyMap[model], taggedDepFile, configFile, dictFile, WMParamFile, OMParamFile, patternFile, negFile, resultFile)
            elif model == 'WM_OLDM_OM':
                cmd = "python3 %s %s %s %s -WM %s -OLDM %s -OM %s -tp %s -ng %s > %s" % (pyMap[model], taggedDepFile, configFile, dictFile, WMParamFile, OLDMParamFile, OMParamFile, patternFile, negFile, resultFile)
            
            
            paramFile = '%s_params.json' % (prefix)
            cmd += '; echo "%s" >> %s; python3 CollectResult.py testScore %s %s 0 6 >> %s' % (prefix, scoreFile, resultFile, paramFile, scoreFile)
            cmd = 'echo "%s" >> %s.1; python3 CollectResult.py testScore %s %s 0 6 >> %s' % (prefix, scoreFile, resultFile, paramFile, scoreFile)
            os.system(cmd)
            print(cmd)
            #sender.putTask(cmd)
            cnt += 1
    
    
    for model in ['WM', 'OLDM', 'OM', 'WM_OLDM', 'WM_OM', 'WM_OLDM_OM']:
    #for model in ['WM', 'OLDM', 'OM']:
    #for model in ['WM_OLDM']:
        for fSelect in ['L1C1', 'RF']:
            prefix = '%s_std_%s' % (model, fSelect)
            configFile = configFolder + '%s_config.json' % (prefix)
            resultFile = '%s_results.csv' %(prefix)
            WMParamFile = 'WM_std_%s_params.json' % (fSelect)
            OLDMParamFile = 'OLDM_std_%s_params.json' % (fSelect)
            OMParamFile = 'OM_std_%s_params.json' % (fSelect)
            if model == 'WM':
                cmd = "python3 %s %s %s > %s" % (pyMap[model], taggedFile, configFile, resultFile)
            elif model == 'OLDM':
                cmd = "python3 %s %s %s %s > %s" %(pyMap[model], depFile, configFile, dictFile, resultFile) 
            elif model == 'OM':
                cmd = "python3 %s %s %s %s %s %s > %s" % (pyMap[model], depFile, configFile, patternFile, negFile, dictFile, resultFile)
            elif model == 'WM_OLDM':
                cmd = "python3 %s %s %s %s -WM %s -OLDM %s > %s" % (pyMap[model], taggedDepFile, configFile, dictFile, WMParamFile, OLDMParamFile, resultFile)
            elif model == 'WM_OM':
                cmd = "python3 %s %s %s %s -WM %s -OM %s -tp %s -ng %s > %s" % (pyMap[model], taggedDepFile, configFile, dictFile, WMParamFile, OMParamFile, patternFile, negFile, resultFile)
            elif model == 'WM_OLDM_OM':
                cmd = "python3 %s %s %s %s -WM %s -OLDM %s -OM %s -tp %s -ng %s > %s" % (pyMap[model], taggedDepFile, configFile, dictFile, WMParamFile, OLDMParamFile, OMParamFile, patternFile, negFile, resultFile)
            
            paramFile = '%s_params.json' % (prefix)
            #cmd += '; echo "%s" >> %s; python3 CollectResult.py testScore %s %s 0 6 >> %s' % (prefix, scoreFile, resultFile, paramFile, scoreFile)
            cmd = 'echo "%s" >> %s.1; python3 CollectResult.py testScore %s %s 0 6 >> %s' % (prefix, scoreFile, resultFile, paramFile, scoreFile)
            os.system(cmd)
            print(cmd)
            #sender.putTask(cmd)
            cnt += 1
    print(cnt)

