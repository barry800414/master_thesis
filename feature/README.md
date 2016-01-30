
## Topic Table
Original New   Name 
2              核四
3         1    服貿
4         2    自經
5         3    美牛
13        4    基薪

## for generating basic feature
python3 BasicFeature.py ../data/taggedLabelNews_5T_Merged_long.json 2 docLen docLen
python3 BasicFeature.py ../data/taggedLabelNews_5T_Merged_long.json 2 nUniqueWord nUniqueWord
python3 BasicFeature.py ../data/taggedLabelNews_5T_Merged_long.json 2 wordDiv wordDiv
python3 BasicFeature.py ../data/taggedLabelNews_5T_Merged_long.json 2 groupCnt groupCnt ../res/SD_large_new.volc

## for generating dependency features
for t in "PP" "PPAll" "POS" "POSAll" "Full" "FullAll" "stance"; do python3 DepFeature.py ../data/DepParsedLabelNews_5T_Merged_withNoLabel_long.json ../config/Dep_${t}.json ../res/negPattern.json ../res/NTUSD_core.csv; done

for t in "PP" "PPAll" "POS" "POSAll" "Full" "FullAll" "stance"; do for i in 3 4 5 13; do python3 ../dimReduction/dimReduction.py t${i}_Dep_${t}.pickle tfidf 0 -method df -minCnt 2 -outPickle t${i}_Dep_${t}_df2.pickle; done; done

## for generating baseline (PP)


## for generating baseline2 (PPAll)
for i in 2 3 4 5 13; do python3 Merge.py t${i}_baseline2.pickle t${i}_BOW_tf.pickle t${i}_Dep_FullAll.pickle t${i}_Dep_POSAll.pickle t${i}_Dep_PPAll.pickle; done
for i in 2 3 4 5 13; do python3 ../dimReduction/dimReduction.py t${i}_baseline2.pickle tfidf 0 -method df -minCnt 3 -outPickle t${i}_baseline2_df3.pickle -noRun; done

## for generating doc-sentence features
python3 WordSentFeature.py ../data/taggedLabelNews_5T_Merged_withNoLabel_long.json ../config/WM_sent_config.json ../res/SD_large_new.volc 3


## merging best individual features
BOW: 59, 23, 26, 19
DepPP: 37, 4, 7, 12
DepFull: 30, 6, 4, 10
DepPOS: 8, 10, 10, 8


BOW: 59, 23, 26, 19
DepPPAll: 30, 12, 9, 8
DepFullAll: 7, 23, 9, 7
DepPOSAll: 30, 12, 12, 17


# for generating shareVolc features (baseline1 and baseline2), all of them are df2. Output: tX_v1_shareVolc.pickle and tX_v2_shareVolc.pickle
for i in 3 4 5 13; do python3 Merge.py t${i}_v1_shareVolc.pickle 0 ./BOW_tf_shareVolc/t${i}_BOW_tf_shareVolc.pickle ./Dep_PP_shareVolc/t${i}_Dep_PP_shareVolc.pickle ./Dep_Full_shareVolc/t${i}_Dep_Full_shareVolc.pickle ./Dep_POS_shareVolc/t${i}_Dep_POS_shareVolc.pickle; done
for i in 3 4 5 13; do python3 Merge.py t${i}_v2_shareVolc.pickle 0 ./BOW_tf_shareVolc/t${i}_BOW_tf_shareVolc.pickle ./Dep_PPAll_shareVolc/t${i}_Dep_PPAll_shareVolc.pickle ./Dep_FullAll_shareVolc/t${i}_Dep_FullAll_shareVolc.pickle ./Dep_POSAll_shareVolc/t${i}_Dep_POSAll_shareVolc.pickle; done


## for merging combination (word+2word+3word+dep_pos+dep_pp+dep_full)
for i in 3 4 5 13; do python3 Merge.py t${i}_merge1_df2.pickle 1 ./BOW_tf/t${i}_BOW_tf_df2.pickle ./2Word/t${i}_2Word_df2.pickle ./3Word/t${i}_3Word_df2.pickle ./Dep_Full/t${i}_Dep_Full_df2.pickle ./Dep_POS/t${i}_Dep_POS_df2.pickle ./Dep_PP/t${i}_Dep_PP_df2.pickle ; done
for i in 3 4 5 13; do python3 Merge.py t${i}_merge2_df2.pickle 1 ./BOW_tf/t${i}_BOW_tf_df2.pickle ./2Word/t${i}_2Word_df2.pickle ./3Word/t${i}_3Word_df2.pickle ./Dep_FullAll/t${i}_Dep_FullAll_df2.pickle ./Dep_POSAll/t${i}_Dep_POSAll_df2.pickle ./Dep_PPAll/t${i}_Dep_PPAll_df2.pickle ; done

## for merging combination (word+2word+3word+dep_pos+dep_pp+dep_full)
for i in 3 4 5 13; do python3 Merge.py t${i}_merge1_shareVolc.pickle 1 ./BOW_tf_shareVolc/t${i}_BOW_tf_shareVolc.pickle ./2Word_shareVolc/t${i}_2Word_shareVolc.pickle ./3Word_shareVolc/t${i}_3Word_shareVolc.pickle ./Dep_Full_shareVolc/t${i}_Dep_Full_shareVolc.pickle ./Dep_POS_shareVolc/t${i}_Dep_POS_shareVolc.pickle ./Dep_PP_shareVolc/t${i}_Dep_PP_shareVolc.pickle ; done
for i in 3 4 5 13; do python3 Merge.py t${i}_merge2_shareVolc.pickle 1 ./BOW_tf_shareVolc/t${i}_BOW_tf_shareVolc.pickle ./2Word_shareVolc/t${i}_2Word_shareVolc.pickle ./3Word_shareVolc/t${i}_3Word_shareVolc.pickle ./Dep_FullAll_shareVolc/t${i}_Dep_FullAll_shareVolc.pickle ./Dep_POSAll_shareVolc/t${i}_Dep_POSAll_shareVolc.pickle ./Dep_PPAll_shareVolc/t${i}_Dep_PPAll_shareVolc.pickle ; done

