#!/bin/bash
tail -n 1 *.log
grep "Error" *.log
python3 showTaskNum.py ${1}
