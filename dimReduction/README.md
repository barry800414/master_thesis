
tfidf.py: tf, tfidf, df
LDA.py: LDA
RunExperiments.py: feature selection
llda.py: Labeled LDA
    input: labels: 2d-list, labels[i] is the list of labels of i-th doc
           corpus: 2d-list, corpus[i] is the list of words of i-th doc
    output: theta, the document-topic distribution 2d numpy array (#doc x #topics)


cmd:
for i in 2 3 4 5 13; do python3 dimReduction.py ../feature/t${i}_BOW_tf.pickle tfidf 0 -method df -minCnt 5 -outPickle ../feature/t${i}_BOW_tf_dfmin5.pickle -noRun ; done


## dimension reduction on multi-level data
python3 dimReductionSent.py ../feature/t3_BOW_sent_tf.pickle 10 10 1 test.pickle
