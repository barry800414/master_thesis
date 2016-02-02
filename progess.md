
2016/01/07
Todo:
2. Should compare the results if using K-Means algorithm
    **see README.md in ./classifier folder 
    Should run: 
      1.DFM.py: Direct feature merging (using community detection)
      2.FM.py: two-side feature merging (using community detection)
      3.DFM_KMeans: Direct feature merging (using K-Means)
      4.FM_KMeans.py: two-side feature merging (using K-Means)

    on features:
      1. each single features
      2. merged features

    To compare: 
      1. DFM & DFM_KMeans  
      2. FM & FM_KMeans

    There should be 10 results: 1.Direct / InDirect  2. Community / KMeans  3. single / Merged
    1. Direct Community Single (done)
    2. Direct Community Merge (done)
    3. Direct KMeans Single (done)
    4. Direct KMeans Merge (done)
    5. InDirect Community Single (done)
    6. InDirect Community Merge (done)
    7. InDirect KMeans Single (can try, it will be very slow, running)
    8. InDirect KMeans Merge

    Summary: KMeans is good, but it is too slow for direct feature merging, and direct feature merging can slightly better
    Future work: Faster KMeans

3. Should investigate the sensitivity of threshold selection in community detection
    **see README.md in ./ResultAnalysis folder
    Should run:
      1. PlotThreshold.py: convert data to excel friendly format
      2. ResultStat.py: calculating statistics about the result

    Summary: try to explain it is not sensitive, and general larger is better?

4. Should inverstigate the physical meaning when merging features
    to do: model (how features are clustered and merged has been dumped), now is to analyze it 
    Should run: ./ResultAnalysis/PrintFeatureCluster.py 

5. Should explain why Community detection can win K-Means ?
    Actually K-Means is too slow (to explain), try other faster clustering algorithm 
    TODO: 
        1.find faster clustering algorithm & explain Kmeans is too slow
            time complexity of KMeans: O(knT)
            other time complexity: 
            Lourvain method: ?

6.  TODO: to find the list of "opinion operator", there are 3 papers to check

7.  Divide (+1/-1) into 2 groups, instead of multiply it by -1
    TODO: 
        1. modify the codes for calculating auxiliary vectors (for Dependency features), and the way to divide groups (done)
        2. for FM and DFM (using community detection), generates new adjacency lists for version 2
        3. rerun FM and DFM for version 2 
        4. for FM_KMeans and DFM_KMeans, rerun all experiments for version 2

10. discuss mult-task


