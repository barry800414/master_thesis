#!/bin/bash
python3 showTaskNum.py
grep "Error" *.log
tail -n 1 *.log

