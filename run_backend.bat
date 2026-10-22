@echo off
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%~dp0backend\app"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir "%~dp0backend\app"
