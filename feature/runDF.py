
import os

for t in [2, 3, 4, 5, 13]:
    for f in ['Dep_PP', 'Dep_PPAll', 'Dep_Full', 'Dep_FullAll', 'Dep_POS', 'Dep_POSAll']:
        cmd = 'python3 ../dimReduction/dimReduction.py ./%s/t%d_%s.pickle 1 tfidf 0 -method df -minCnt 2 -outPickle ./%s/t%d_%s_df2.pickle -noRun' % (f, t, f, f, t, f)
        print(cmd)
        os.system(cmd)

