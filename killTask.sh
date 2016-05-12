
ps aux | grep client.py | awk '{print $2}' | xargs kill -9
ps aux | grep server.py | awk '{print $2}' | xargs kill -9 
ps aux | grep FM.py | awk '{print $2}' | xargs kill -9
