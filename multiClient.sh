
if [ "$#" -ne 3 ]; then
    echo "./multiClient.sh nodeId #client port"
else
    for (( i=1; i<=${2}; i++ ))
    do
        python3 client.py ${3} 2> ${1}_${i}.log &
    done
fi
