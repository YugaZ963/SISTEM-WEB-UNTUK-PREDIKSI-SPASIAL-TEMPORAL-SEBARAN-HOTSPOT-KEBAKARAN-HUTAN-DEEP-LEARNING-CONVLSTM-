# File anotasi dari `run_train_native_patch_wsl_gpu.sh`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pada script terminal.
# Komentar file skripsi: Script WSL untuk menjalankan training native patch ConvLSTM dengan GPU.
# Catatan pada script terminal.
# Konteks laporan: mendukung proses eksperimen/training pada BAB IV.

# Menentukan program yang menjalankan script ini.
#!/usr/bin/env bash
# Mengatur cara script terminal dijalankan.
set -euo pipefail

# Menjalankan perintah terminal untuk kebutuhan project.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Menjalankan perintah terminal untuk kebutuhan project.
VENV_DIR="${WSL_TF_VENV:-$HOME/.venvs/sistem-web-skripsi-ta}"

# Menjalankan script Python untuk training, evaluasi, atau dokumentasi.
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  # Menjalankan perintah terminal untuk kebutuhan project.
  echo "Virtual environment WSL tidak ditemukan: $VENV_DIR" >&2
  # Menjalankan perintah terminal untuk kebutuhan project.
  echo "Buat dulu dengan:" >&2
  # Menjalankan script Python untuk training, evaluasi, atau dokumentasi.
  echo "  python3 -m venv $VENV_DIR" >&2
  # Menjalankan perintah terminal untuk kebutuhan project.
  echo "  source $VENV_DIR/bin/activate" >&2
  # Menjalankan script Python untuk training, evaluasi, atau dokumentasi.
  echo "  python -m pip install --upgrade pip setuptools wheel" >&2
  # Menjalankan script Python untuk training, evaluasi, atau dokumentasi.
  echo "  python -m pip install \"tensorflow[and-cuda]\"" >&2
  # Menjalankan perintah terminal untuk kebutuhan project.
  exit 1
# Menjalankan perintah terminal untuk kebutuhan project.
fi

# Menjalankan script Python untuk training, evaluasi, atau dokumentasi.
SITE_PACKAGES="$("$VENV_DIR/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
# Menjalankan perintah terminal untuk kebutuhan project.
NVIDIA_ROOT="$SITE_PACKAGES/nvidia"

# Menjalankan perintah terminal untuk kebutuhan project.
if [[ ! -d "$NVIDIA_ROOT" ]]; then
  # Menjalankan perintah terminal untuk kebutuhan project.
  echo "Folder library NVIDIA dari pip tidak ditemukan: $NVIDIA_ROOT" >&2
  # Menjalankan perintah terminal untuk kebutuhan project.
  echo "Install TensorFlow GPU dulu dengan:" >&2
  # Menjalankan perintah terminal untuk kebutuhan project.
  echo "  source $VENV_DIR/bin/activate" >&2
  # Menjalankan script Python untuk training, evaluasi, atau dokumentasi.
  echo "  python -m pip install \"tensorflow[and-cuda]\"" >&2
  # Menjalankan perintah terminal untuk kebutuhan project.
  exit 1
# Menjalankan perintah terminal untuk kebutuhan project.
fi

# Menjalankan perintah terminal untuk kebutuhan project.
export LD_LIBRARY_PATH="$NVIDIA_ROOT/cublas/lib:$NVIDIA_ROOT/cuda_runtime/lib:$NVIDIA_ROOT/cudnn/lib:$NVIDIA_ROOT/cufft/lib:$NVIDIA_ROOT/curand/lib:$NVIDIA_ROOT/cusolver/lib:$NVIDIA_ROOT/cusparse/lib:$NVIDIA_ROOT/nccl/lib:$NVIDIA_ROOT/nvjitlink/lib:$NVIDIA_ROOT/cuda_nvrtc/lib:$NVIDIA_ROOT/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"
# Menjalankan perintah terminal untuk kebutuhan project.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

# Menjalankan perintah terminal untuk kebutuhan project.
cd "$PROJECT_ROOT"

# Menjalankan script Python untuk training, evaluasi, atau dokumentasi.
exec "$VENV_DIR/bin/python" Ipynb/TRAIN_CONVLSTM_NATIVE_PATCH_1528x773_GOOGLE_COLAB.py "$@"
