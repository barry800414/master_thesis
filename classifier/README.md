
## run baseline (not PPAll)
for i in 2 3 4 5 13; do echo "run t${i}"; python3 Run.py ../feature/t${i}_baseline_df3.pickle 3 > ./baseline/t${i}_baseline_df3_results.csv; done
for i in 2 3 4 5 13; do python3 ../CollectResult.py ./baseline/t${i}_baseline_df3_results.csv >> baseline_result.csv ; done

## generating sentence-level (svm-sle) data


## run merged
for i in 3 4 5 13; do python3 ./Run.py ../feature/t${i}_v1.pickle 3 > ./merged/t${i}_v1_results.csv ; done
for i in 3 4 5 13; do python3 ../CollectResult.py ./merged/t${i}_v1_results.csv >> merged_result.csv; done
