
tfidf.py: tf, tfidf, df
LDA.py: LDA
RunExperiments.py: feature selection
llda.py: Labeled LDA
    input: labels: 2d-list, labels[i] is the list of labels of i-th doc
           corpus: 2d-list, corpus[i] is the list of words of i-th doc
    output: theta, the document-topic distribution 2d numpy array (#doc x #topics)


cmd:
for i in 2 3 4 5 13; do python3 dimReduction.py ../feature/BOW/t${i}_BOW_tf.pickle tfidf 0 -method df -minCnt 2 -outPickle ../feature/BOW/t${i}_BOW_tf_df2.pickle ; done

## dimension reduction on multi-level data
python3 dimReductionSent.py ../feature/t3_BOW_sent_tf.pickle 22 500 0 ./sentLDA/t3_baseline_nT22_nI500_ux0.pickle
python3 dimReductionSent.py ../feature/t4_BOW_sent_tf.pickle 9 500 0 ./sentLDA/t4_baseline_nT9_nI500_ux0.pickle
python3 dimReductionSent.py ../feature/t5_BOW_sent_tf.pickle 13 500 0 ./sentLDA/t5_baseline_nT13_nI500_ux0.pickle
python3 dimReductionSent.py ../feature/t13_BOW_sent_tf.pickle 16 500 0 ./sentLDA/t13_baseline_nT16_nI500_ux0.pickle

## dimension reduction (LDA) on share-volc data (for multi-task learning)
python3 dimReductionShareVolc.py LDA 0 LDA_shareVolc -inPickle t3 ../feature/t3_BOW_tf_shareVolc.pickle -inPickle t4 ../feature/t4_BOW_tf_shareVolc.pickle -inPickle t5 ../feature/t5_BOW_tf_shareVolc.pickle -inPickle t13 ../feature/t13_BOW_tf_shareVolc.pickle -nTopics 30 -nIter 500


## dimension reduction (PCA) on share-volc data (for multi-task learning)



## dimension reduction (LDA) on sentence level data (for multi-level method)
for i in 3 4 5 13; do python3 dimReductionSent2.py ../feature/t${i}_BOW_sent_tf.pickle 1 500 0 ../feature/t${i}_BOW_sent_tf_LDA1.pickle; done
