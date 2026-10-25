@echo off
echo Starting Smart Analysis & Investment System...
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m streamlit run app.py
pause
