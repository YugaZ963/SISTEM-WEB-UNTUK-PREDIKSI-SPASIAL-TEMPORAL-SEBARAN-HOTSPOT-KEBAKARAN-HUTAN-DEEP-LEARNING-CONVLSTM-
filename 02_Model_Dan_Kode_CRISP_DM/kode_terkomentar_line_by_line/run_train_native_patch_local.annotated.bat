REM File anotasi dari `run_train_native_patch_local.bat`.
REM File ini untuk belajar/lampiran, bukan source utama aplikasi.
REM Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM.
REM Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
REM Catatan pada script Windows.
REM Komentar file skripsi: Script Windows untuk menjalankan training native patch secara lokal.
REM Catatan pada script Windows.
REM Konteks laporan: mendukung proses eksperimen/training pada BAB IV.

REM Mengatur apakah perintah ditampilkan saat script berjalan.
@echo off
REM Menjalankan perintah Windows untuk kebutuhan project.
setlocal

REM Menjalankan perintah Windows untuk kebutuhan project.
cd /d "%~dp0"

REM Menjalankan script Python lewat file Windows.
if not exist ".venv\Scripts\python.exe" (
  REM Menjalankan script Python lewat file Windows.
  echo ERROR: File .venv\Scripts\python.exe tidak ditemukan.
  REM Menjalankan perintah Windows untuk kebutuhan project.
  echo Pastikan virtual environment .venv sudah ada di folder project ini.
  REM Menjalankan perintah Windows untuk kebutuhan project.
  exit /b 1
REM Menjalankan perintah Windows untuk kebutuhan project.
)

REM Menjalankan script Python lewat file Windows.
".venv\Scripts\python.exe" "Ipynb\TRAIN_CONVLSTM_NATIVE_PATCH_1528x773_GOOGLE_COLAB.py" %*

REM Menjalankan perintah Windows untuk kebutuhan project.
endlocal
