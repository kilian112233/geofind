@echo off
cd /d W:\geofind
"C:\Users\Windows\AppData\Local\Programs\Python\Python313\python.exe" dev\build_retrieval_db.py --force --limit 5000 > dev\build_db.log 2>&1
