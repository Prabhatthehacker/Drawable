@echo off
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 goto nopython
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
nvidia-smi >nul 2>&1
if errorlevel 1 goto cpu
pip install torch --index-url https://download.pytorch.org/whl/cu121
goto rest
:cpu
pip install torch
:rest
pip install -r requirements.txt
python -c "from diffusers import AutoPipelineForText2Image as A; A.from_pretrained('stabilityai/sd-turbo')"
echo.
echo Setup complete. Double-click check.bat to verify, then run.bat to start.
echo If torch fails with WinError 126, install https://aka.ms/vs/17/release/vc_redist.x64.exe and restart.
pause
exit /b
:nopython
echo Install Python 3.10 or newer from python.org first (tick "Add Python to PATH"), then run setup.bat again.
pause
