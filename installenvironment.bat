call py -3.10 -m venv "%~2MoDL"
c:
cd "%~2MoDL\Scripts\" 
call activate.bat
cd "%1"
call python.exe -m pip install -r requirements.txt
PAUSE