# File anotasi dari `backend/settings.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Konfigurasi utama model, path storage, profil historical_risk_patch_160, dan visualisasi web.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import os
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
BASE_DIR = Path(__file__).resolve().parent
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
PROJECT_ROOT = BASE_DIR.parent
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
STORAGE_DIR = BASE_DIR / "storage"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
INPUTS_DIR = STORAGE_DIR / "inputs"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
OUTPUTS_DIR = STORAGE_DIR / "outputs"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ASSETS_DIR = BASE_DIR / "assets"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
STATIC_DIR = BASE_DIR / "static"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
TEMPLATES_DIR = BASE_DIR / "templates"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
MODELS_DIR = BASE_DIR / "models"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
IPYNB_DIR = PROJECT_ROOT / "Ipynb"

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
BASE_MAP_CANDIDATES = [
    # Melanjutkan langkah kerja pada bagian kode ini.
    PROJECT_ROOT / "peta_riau.jpg",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ASSETS_DIR / "peta_riau.jpg",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ASSETS_DIR / "base_map.png",
# Menutup susunan data atau perintah yang dimulai sebelumnya.
]
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
BASE_MAP_PATH = next((path for path in BASE_MAP_CANDIDATES if path.exists()), ASSETS_DIR / "base_map.png")
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
CONVLSTM_MODEL_FILE = MODELS_DIR / "convlstm_model.h5"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
CONVLSTM_SAVEDMODEL_DIR = MODELS_DIR / "convlstm_saved_model"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
CONVLSTM_IPYNB_KERAS_FILE = IPYNB_DIR / "best_model_convlstm.keras"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
CONVLSTM_IPYNB_H5_FILE = IPYNB_DIR / "best_model_convlstm.h5"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
HISTORICAL_RISK_MODEL_DIR = MODELS_DIR / "historical_risk_patch_20260416"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
HISTORICAL_RISK_MODEL_FILE = HISTORICAL_RISK_MODEL_DIR / "best_model_convlstm_historical_risk_patch.keras"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
HISTORICAL_RISK_REPORT_FILE = HISTORICAL_RISK_MODEL_DIR / "training_report_best_1epoch.json"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
HISTORICAL_RISK_EPOCH3_REPORT_FILE = HISTORICAL_RISK_MODEL_DIR / "training_report_epoch3_comparison.json"

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
LEGACY_CONVLSTM_MODEL_CANDIDATES = [
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    CONVLSTM_MODEL_FILE,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    CONVLSTM_SAVEDMODEL_DIR,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    CONVLSTM_IPYNB_KERAS_FILE,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    CONVLSTM_IPYNB_H5_FILE,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
]

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
MODEL_PROFILE_CONFIGS = {
    # Melanjutkan langkah kerja pada bagian kode ini.
    "historical_risk_patch_160": {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "display_name": "Historical Risk Patch ConvLSTM 160x160",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "description": "Model patch 160x160 untuk prediksi area risiko hotspot berbasis citra historis.",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_candidates": [HISTORICAL_RISK_MODEL_FILE],
        # Melanjutkan langkah kerja pada bagian kode ini.
        "training_report": HISTORICAL_RISK_REPORT_FILE,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "comparison_report": HISTORICAL_RISK_EPOCH3_REPORT_FILE,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "grid_size": 160,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "channels": 1,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "time_steps": 7,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "preprocess_mode": "hotspot_red_mask",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "inference_mode": "patch_stitch",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_size": 160,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_stride": 80,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_batch_size": 32,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_dilation_kernel": 5,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recommended_threshold": 0.55,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "discover_ipynb_models": False,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    },
    # Melanjutkan langkah kerja pada bagian kode ini.
    "legacy_128": {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "display_name": "Legacy ConvLSTM 128x128",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "description": "Model lama 128x128 dengan resize langsung seluruh citra.",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_candidates": LEGACY_CONVLSTM_MODEL_CANDIDATES,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "training_report": None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "comparison_report": None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "grid_size": 128,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "channels": 1,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "time_steps": 7,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "preprocess_mode": "hotspot_red_mask",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "inference_mode": "direct_resize",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_size": None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_stride": None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_batch_size": None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_dilation_kernel": 3,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recommended_threshold": 0.5,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "discover_ipynb_models": True,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    },
# Menutup susunan data atau perintah yang dimulai sebelumnya.
}

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ACTIVE_MODEL_PROFILE = os.getenv("HOTSPOT_MODEL_PROFILE", "historical_risk_patch_160").strip()
# Mengecek syarat sebelum melanjutkan proses.
if ACTIVE_MODEL_PROFILE not in MODEL_PROFILE_CONFIGS:
    # Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
    ACTIVE_MODEL_PROFILE = "historical_risk_patch_160"

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ACTIVE_MODEL_CONFIG = MODEL_PROFILE_CONFIGS[ACTIVE_MODEL_PROFILE]
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
CONVLSTM_MODEL_CANDIDATES = ACTIVE_MODEL_CONFIG["model_candidates"]

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
REQUIRED_STEMS = ["H-6", "H-5", "H-4", "H-3", "H-2", "H-1", "H0"]
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
MAX_FILE_MB = 5
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_GRID_SIZE = 128
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_CHANNELS = 1
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_TIME_STEPS = 7
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
OVERLAY_MIN_VISIBLE_PROBABILITY = 0.15
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ADMIN_BOUNDARY_GEOJSON_PATH = PROJECT_ROOT / "data-admin" / "batas_kabupaten_kota_riau_geoboundaries_adm2.geojson"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ADMIN_BOUNDARY_OVERLAY_ENABLED = os.getenv("HOTSPOT_OVERLAY_ADMIN_BOUNDARY", "1").strip().lower() not in {
    # Melanjutkan langkah kerja pada bagian kode ini.
    "0",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "false",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "no",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "off",
# Menutup susunan data atau perintah yang dimulai sebelumnya.
}
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ADMIN_BOUNDARY_CENTER_LON = float(os.getenv("HOTSPOT_MAP_CENTER_LON", "102.1"))
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ADMIN_BOUNDARY_CENTER_LAT = float(os.getenv("HOTSPOT_MAP_CENTER_LAT", "0.8"))
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ADMIN_BOUNDARY_ZOOM = float(os.getenv("HOTSPOT_MAP_ZOOM", "8.1"))
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ADMIN_BOUNDARY_LINE_WIDTH = int(os.getenv("HOTSPOT_ADMIN_BOUNDARY_LINE_WIDTH", "3"))
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ADMIN_BOUNDARY_OUTLINE_WIDTH = int(os.getenv("HOTSPOT_ADMIN_BOUNDARY_OUTLINE_WIDTH", "5"))


# Membuat langkah kerja bernama `initialize_directories`.
def initialize_directories() -> None:
    # Mengulang proses untuk setiap data dalam daftar.
    for path in [
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        STORAGE_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        INPUTS_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        OUTPUTS_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        ASSETS_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        STATIC_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        TEMPLATES_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        MODELS_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        HISTORICAL_RISK_MODEL_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        IPYNB_DIR,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ]:
        # Menyimpan nilai ke `path.mkdir(parents` untuk dipakai pada langkah berikutnya.
        path.mkdir(parents=True, exist_ok=True)
