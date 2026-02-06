@echo off
call conda activate videosubx
if %errorlevel% neq 0 (
    echo Failed to activate conda environment 'videosubx'.
    pause
    exit /b
)

echo Starting VideoSubX WebUI...
python -m uvicorn main:app --host 0.0.0.0 --port 8501 --reload

pause
