@echo off
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
  echo The venv folder is missing. Double-click setup.bat first.
  pause
  exit /b
)
call venv\Scripts\activate
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
python paint_ai.py
pause
