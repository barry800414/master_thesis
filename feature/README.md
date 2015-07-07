
## for generating basic feature
python3 BasicFeature.py ../data/taggedLabelNews_5T_Merged_long.json 2 docLen docLen
python3 BasicFeature.py ../data/taggedLabelNews_5T_Merged_long.json 2 nUniqueWord nUniqueWord
python3 BasicFeature.py ../data/taggedLabelNews_5T_Merged_long.json 2 wordDiv wordDiv
python3 BasicFeature.py ../data/taggedLabelNews_5T_Merged_long.json 2 groupCnt groupCnt ../res/SD_large_new.volc

## for generating baseline (PP)

## for generating baseline2 (PPAll)
for i in 2 3 4 5 13; do python3 Merge.py t${i}_baseline2.pickle t${i}_BOW_tf.pickle t${i}_Dep_FullAll.pickle t${i}_Dep_POSAll.pickle t${i}_Dep_PPAll.pickle; done
for i in 2 3 4 5 13; do python3 ../dimReduction/dimReduction.py t${i}_baseline2.pickle tfidf 0 -method df -minCnt 3 -outPickle t${i}_baseline2_df3.pickle -noRun; done

## for generating doc-sentence features
python3 WordSentFeature.py ../data/taggedLabelNews_5T_Merged_withNoLabel_long.json ../config/WM_sent_config.json ../res/SD_large_new.volc 3

