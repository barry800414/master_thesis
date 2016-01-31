
This folder contains all the necessary files of my master thesis. 

## Introduction of each folder
    ./classifier: implement all the algorithms
    ./feature: all the features extracted from raw data (most of them are pickle files)
    ./res: all pattern files for extracting dependency features, dictionaries.
    ./data: all raw data and labels in different forms
    ./errorAnalysis: implement the functions to do error analysis

    ./ResultAnalysis: implement all the functions to analyze results

## Collect Results
* ?
    python3 CollectResult.py
* Find the best threshold via validation scores
    python3 FindBest.py csv
* Calculate the average and standard deviation of testing scores, find best via validation scores
    python3 ResultStat.py csv
* Re-organize the data into excel friendly format, plot the figures if needed
    python3 PlotThresholdTest.py csv [FigureOutputFolder]

## useful information
### Feature Type Name mapping
    thesis  : file       : fType in codes
    BOW_tf  : BOW        : Word
    2Word   : Bi         : BiWord
    3Word   : Tri        : TriWord
    PT_SB   : Dep_PP     : H_P, T_P, H/T_P, H_N, T_N, H/T_N
    Full_SB : Dep_Full   : OT, HO, HO/OT
    PT_TB   : Dep_PPAll  : H_P, T_P, H/T_P, H_N, T_N, H/T_N
    Full_TB : Dep_FullAll: OT, HO, HO/OT

    *(Full = full representation)
    *(all = TB tag based)
