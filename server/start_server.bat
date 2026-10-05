@echo off
title SurfShield Server
cd /d "%~dp0"
echo ============================================
echo   SurfShield Server - Starting on port 8000
echo ============================================
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
