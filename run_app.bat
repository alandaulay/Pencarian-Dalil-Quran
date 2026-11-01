@echo off
echo ========================================================
echo   MENJALANKAN APLIKASI PENCARIAN DALIL AL-QUR'AN
echo   Information Retrieval (VSM & TF-IDF) - Streamlit
echo ========================================================
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
