#!/bin/bash
mkdir -p sentLDA

t=4
nT=81
python3 dimReductionSent.py ../feature/t4_BOW_sent_tf.pickle ${nT} 500 0 ./sentLDA/t${t}_BOW_sent_nT${nT}_nI500_ux0.pickle
python3 dimReductionSent.py ../feature/t4_BOW_sent_tf.pickle ${nT} 500 1 ./sentLDA/t${t}_BOW_sent_nT${nT}_nI500_ux1.pickle

t=5
nT=26
python3 dimReductionSent.py ../feature/t4_BOW_sent_tf.pickle ${nT} 500 0 ./sentLDA/t${t}_BOW_sent_nT${nT}_nI500_ux0.pickle
python3 dimReductionSent.py ../feature/t4_BOW_sent_tf.pickle ${nT} 500 1 ./sentLDA/t${t}_BOW_sent_nT${nT}_nI500_ux1.pickle

t=6
nT=19
python3 dimReductionSent.py ../feature/t4_BOW_sent_tf.pickle ${nT} 500 0 ./sentLDA/t${t}_BOW_sent_nT${nT}_nI500_ux0.pickle
python3 dimReductionSent.py ../feature/t4_BOW_sent_tf.pickle ${nT} 500 1 ./sentLDA/t${t}_BOW_sent_nT${nT}_nI500_ux1.pickle

