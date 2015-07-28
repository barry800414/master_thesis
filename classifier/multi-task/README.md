
2015/07/11
Please note that now the data and result folder are from the data with Unlabeled news (volc size is 4xxxx)

2015/07/12
The data of PCANone is without unlabeled news (volc size is 1xxxx, BOW_tf features)
The result on 2015/07/11 has been moved to result20150711


## Usage: OneTestSingleFold (some of data in test topic is testing data, all the other data is training data)
    matlab -r "OneTestSingleFold( './data/LDA_nT10_uX0.mat', 'Logistic_Lasso', 1, 1, 1, 'test' )"
