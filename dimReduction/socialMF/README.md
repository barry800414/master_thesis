
## convert pickle file to Social MF format
python3 convertData.py ../../feature/t3_BOW_tf_dfmin5.pickle -outX t3_BOW_df5.X -outY t3_BOW_df5.y -outNetwork t3_BOW_df5_labelOnly.edgelist labelOnly

## run Social Regularized MF 
python3 SR_MF.py t3_BOW_df5 t3_BOW_df5.X t3_BOW_df5_labelOnly.edgelist t3_BOW_df5.y 10 0.1 0.1 100 0.01 1 10 python3 SR_MF.py t3_BOW_df5 t3_BOW_df5.X t3_BOW_df5_labelOnly.edgelist t3_BOW_df5.y 10 0.1 0.1 100 0.01 1 10
