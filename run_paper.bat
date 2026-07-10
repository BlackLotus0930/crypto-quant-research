@echo off
cd /d C:\Users\heloy\market_foundation_model
set PYTHONUTF8=1
.venv\Scripts\python.exe paper_live.py >> data\paper\cron.log 2>&1
