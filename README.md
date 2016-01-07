
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

