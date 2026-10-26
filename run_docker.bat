@echo off
TITLE Bank Muscat P_137 - Docker Runner
echo ====================================================
echo Cleaning up any old containers holding ports...
echo ====================================================

:: Automatically stop and remove any existing container
docker stop bank_muscat_container 2>nul
docker rm bank_muscat_container 2>nul

echo [INFO] Pulling latest image from Docker Hub...
docker pull yash9019/bank-muscat-p137:latest

echo [INFO] Launching container with Docker env-file...
echo ----------------------------------------------------
echo   Streamlit Portal : http://localhost:7860
echo   FastAPI Swagger  : http://localhost:8000/docs
echo ----------------------------------------------------
echo Press Ctrl+C at any time to stop the container.
echo ====================================================

docker run --name bank_muscat_container --rm -p 8000:8000 -p 8501:8501 -p 7860:7860 --env-file .env yash9019/bank-muscat-p137:latest

pause