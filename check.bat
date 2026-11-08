@echo off
cd /d "%~dp0"
call venv\Scripts\activate
echo.
echo === Are the libraries installed? ===
python -c "import importlib.util as u;[print(('OK       ' if u.find_spec(m) else 'MISSING  ')+m) for m in ['tkinter','numpy','cv2','PIL','pyautogui','pygetwindow','torch','diffusers','transformers','accelerate']]"
echo.
echo === Do they load? (GPU shows True/False) ===
python -c "import torch, cv2, diffusers, pyautogui; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('opencv', cv2.__version__); print('diffusers', diffusers.__version__); print('screen', pyautogui.size())"
echo.
echo === Is the image model downloaded? ===
python -c "from pathlib import Path; p=Path.home()/'.cache'/'huggingface'/'hub'/'models--stabilityai--sd-turbo'; print('FOUND' if p.exists() else 'NOT FOUND - run setup.bat')"
echo.
echo === Version conflicts? ===
pip check
echo.
pause
