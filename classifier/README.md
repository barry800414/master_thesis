
This folder contains all the core algorithms 

## Prerequisites 
* numpy
* scipy
* sklearn
* networkx 
* [Louvain method for community detection](https://bitbucket.org/taynaud/python-louvain)

## Folders
* ../feature/: Features pickle file 
* ./featureMerge: all the adjacency list files 

### Important information
         version 1    version 2
    Word      Word         Word
    BiWord    BiWord       BiWord
    TriWord   TriWord      TriWord
    T & 0     H/T_N        H/T_0
    T & 1     H/T_P        H/T_P
    T &-1     H/T_P        H/T_N
    H & 0     H/T_N        H/T_0
    H & 1     H/T_P        H/T_P
    H &-1     H/T_P        H/T_N
    HO& 1     HO/OT        HO/OT_P
    HO&-1     HO/OT        HO/OT_N
    OT& 1     HO/OT        HO/OT_P
    OT&-1     HO/OT        HO/OT_P
    HOT         X            X
              P=polarity   P=Positive
              N=neutral    N=Negative
                           0=Neutral
    In version 1, positive and negative are put in same group, feature vector is multiplied by -1
    In version 2, positive and negative are put in different groups, feature vector is the same
    Date: 2016/2/1

## Normal training and testing (baseline)
    python3 Run.py pickleFile seedNum [-outLogPickle LogPickle] [--fSelect -method xxx -param1 value1 -param2 value2]
    --fSelect -method chi -top 10
    --fSelect -method RFE -n_features_to_select 100 -step 10 
    --fSelect -method REFCV -step 10 -n_folds 10
    --fSelect -method LinearSVM -C 1.0
    --fSelect -method RF

## Direct Feature clustering (using community detection) firstly, then normal training and testing (Direct Feature Merging, DFM)
    python3 DFM.py pickleFile adjListFile seedNum [-outLogPickle LogPickle] [--fSelect -method xxx -param1 value1 ...] [--preprocess -method xxx -param1 value1 ...]
    
* pickleFile is in ../feature
* adjListFile is in ./featureMerge

## Direct Feature clustering (using K-means) firstly, then normal training and testing (Direct Feature Merging using Kmeans)
    python3 DFM_KMeans.py pickleFile wordVectorFile nClusters seedNum [-outLogPickle LogPickle] [-nClusterFile jsonFile] [--fSelect -method xxx -param1 value1 ...] [--preprocess -method xxx -param1 value1 ...]

## Feature clustering (using community detection) in each train&test phase. (Our proposed solution)
    python3 FM.py pickleFile adjListFile version seedNum [-outLogPickle LogPickle] [--fSelect -method xxx -param1 value1 ...] [--preprocess -method xxx -param1 value1 ...]

* pickleFile is in ../feature
* adjListFile is in ./featureMerge

## Feature clustering (using K-Means) in each train&test phase.
    python3 FM_KMeans.py pickleFile wordVectorFile nClusters seedNum [-outLogPickle LogPickle] [--preprocess -method xxx -param1 value1 ...]

* wordVectorFile is in ./



# Classifier Usage (all train one test framework)
    OneTestRun.py topic seedNum [-topic1 pickle -topic2 pickle ...] [-outLogPickle LogPickle] [--fSelect -method xxx -param1 value1 -param2 value2]
    

## run baseline (not PPAll)
for i in 2 3 4 5 13; do echo "run t${i}"; python3 Run.py ../feature/t${i}_baseline_df3.pickle 3 > ./baseline/t${i}_baseline_df3_results.csv; done
for i in 2 3 4 5 13; do python3 ../CollectResult.py ./baseline/t${i}_baseline_df3_results.csv >> baseline_result.csv ; done

## generating sentence-level (svm-sle) data


## run merged
for i in 3 4 5 13; do python3 ./Run.py ../feature/t${i}_v1.pickle 3 > ./merged/t${i}_v1_results.csv ; done
for i in 3 4 5 13; do python3 ../CollectResult.py ./merged/t${i}_v1_results.csv >> merged_result.csv; done

