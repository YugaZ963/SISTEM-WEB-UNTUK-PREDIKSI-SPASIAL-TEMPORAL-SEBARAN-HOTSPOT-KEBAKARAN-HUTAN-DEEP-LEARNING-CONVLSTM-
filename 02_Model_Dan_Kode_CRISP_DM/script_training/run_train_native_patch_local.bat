REM Komentar file skripsi: Script Windows untuk menjalankan training native patch secara lokal.
REM Konteks laporan: mendukung proses eksperimen/training pada BAB IV.

@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: File .venv\Scripts\python.exe tidak ditemukan.
  echo Pastikan virtual environment .venv sudah ada di folder project ini.
  exit /b 1
)

".venv\Scripts\python.exe" "Ipynb\TRAIN_CONVLSTM_NATIVE_PATCH_1528x773_GOOGLE_COLAB.py" %*

endlocal
