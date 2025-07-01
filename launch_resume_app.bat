@echo off
echo Starting ATS Resume Builder...

REM Set the path to your application folder - replace with your actual path
set APP_PATH=C:\Users\xxx\resume_builder

REM Set your conda environment name - replace with your actual environment name
set CONDA_ENV=project1  

REM Change to the application directory
cd /d %APP_PATH%

REM Activate the conda environment
call conda activate %CONDA_ENV%

REM Open the browser
start "" http://localhost:8000

REM Start the FastAPI server
python -m uvicorn server:app --host 0.0.0.0 --port 8000

REM If the server stops, deactivate the conda environment
call conda deactivate