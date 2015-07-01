
tfidf.py: tf, tfidf, df
LDA.py: LDA
RunExperiments.py: feature selection
llda.py: Labeled LDA
    input: labels: 2d-list, labels[i] is the list of labels of i-th doc
           corpus: 2d-list, corpus[i] is the list of words of i-th doc
    output: theta, the document-topic distribution 2d numpy array (#doc x #topics)
