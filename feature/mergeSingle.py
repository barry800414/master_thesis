
import os


## merging best individual features
tMap = [3, 4, 5, 13]
v1 = { 
    'BOW_tf': [59, 23, 26, 19],
    'Dep_PP': [37, 4, 7, 12],
    'Dep_Full': [30, 6, 4, 10],
    'Dep_POS': [8, 10, 10, 8]
}

for i in [0, 1, 2, 3]:
    cmd = 'python3 Merge.py t%d_v1.pickle 0' % (tMap[i])
    for name in v1.keys():
        cmd += ' ../dimReduction/single_LDA_result/t%d_%s_df2_LDA_nT%d_nI500_uX1.pickle' % (tMap[i], name, v1[name][i])
    print(cmd)
    os.system(cmd)

v2 = {
    'BOW_tf': [59, 23, 26, 19],
    'Dep_PPAll': [30, 12, 9, 8],
    'Dep_FullAll': [7, 23, 9, 7],
    'Dep_POSAll': [30, 12, 12, 17]
}

for i in [0, 1, 2, 3]:
    cmd = 'python3 Merge.py t%d_v2.pickle 0' % (tMap[i])
    for name in v2.keys():
        cmd += ' ../dimReduction/single_LDA_result/t%d_%s_df2_LDA_nT%d_nI500_uX1.pickle' % (tMap[i], name, v2[name][i])
    print(cmd)
    os.system(cmd)


