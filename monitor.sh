#!/bin/bash
python3 showTaskNum.py ${1}
grep "Error" *.log
tail -n 1 *.log

