
if [ "$#" -ne 2 ]; then
    echo "./multiClient.sh cnId #client"
else
    for (( i=1; i<=${2}; i++ ))
    do
        python3 client.py 2> cn${1}_${i}.log &
    done
fi
