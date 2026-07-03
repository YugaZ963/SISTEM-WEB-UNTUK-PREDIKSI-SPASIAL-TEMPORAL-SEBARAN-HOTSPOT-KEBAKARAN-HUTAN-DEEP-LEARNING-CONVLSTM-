"""Komentar file skripsi:
Konfigurasi utama backend untuk path model, threshold, folder storage, dan parameter sistem.

Konteks laporan: file ini mendukung BAB IV pada bagian implementasi/deployment sistem,
khususnya konfigurasi model ConvLSTM, input citra, penyimpanan hasil, dan visualisasi peta risiko hotspot.
"""

# Modul os digunakan untuk membaca environment variable runtime, misalnya pemilihan profile model.
import os

# Path digunakan agar penulisan lokasi file/folder lebih rapi dan lintas sistem operasi.
from pathlib import Path

# Folder dasar backend, yaitu lokasi file settings.py berada.
BASE_DIR = Path(__file__).resolve().parent

# Folder root project, yaitu satu tingkat di atas folder backend.
PROJECT_ROOT = BASE_DIR.parent

# Folder utama untuk penyimpanan data runtime sistem web.
STORAGE_DIR = BASE_DIR / "storage"

# Folder untuk menyimpan citra input H-6 sampai H0 yang diunggah pengguna.
INPUTS_DIR = STORAGE_DIR / "inputs"

# Folder untuk menyimpan output prediksi seperti heatmap, overlay, binary mask, dan metadata.
OUTPUTS_DIR = STORAGE_DIR / "outputs"

# Folder aset pendukung, misalnya peta dasar wilayah Riau.
ASSETS_DIR = BASE_DIR / "assets"

# Folder file static dashboard seperti CSS dan JavaScript.
STATIC_DIR = BASE_DIR / "static"

# Folder template HTML dashboard FastAPI.
TEMPLATES_DIR = BASE_DIR / "templates"

# Folder penyimpanan model machine learning pada backend.
MODELS_DIR = BASE_DIR / "models"

# Folder artefak notebook/eksperimen model dari proses penelitian.
IPYNB_DIR = PROJECT_ROOT / "Ipynb"

# Daftar kandidat peta dasar yang dapat digunakan untuk overlay hasil prediksi.
BASE_MAP_CANDIDATES = [
    PROJECT_ROOT / "peta_riau.jpg",
    ASSETS_DIR / "peta_riau.jpg",
    ASSETS_DIR / "base_map.png",
]

# Memilih peta dasar pertama yang tersedia; jika tidak ada, memakai fallback base_map.png.
BASE_MAP_PATH = next((path for path in BASE_MAP_CANDIDATES if path.exists()), ASSETS_DIR / "base_map.png")

# Lokasi model ConvLSTM lama dalam format H5.
CONVLSTM_MODEL_FILE = MODELS_DIR / "convlstm_model.h5"

# Lokasi model ConvLSTM lama dalam format SavedModel.
CONVLSTM_SAVEDMODEL_DIR = MODELS_DIR / "convlstm_saved_model"

# Lokasi model hasil eksperimen notebook dalam format Keras.
CONVLSTM_IPYNB_KERAS_FILE = IPYNB_DIR / "best_model_convlstm.keras"

# Lokasi model hasil eksperimen notebook dalam format H5.
CONVLSTM_IPYNB_H5_FILE = IPYNB_DIR / "best_model_convlstm.h5"

# Folder model default historical risk patch yang digunakan sistem web.
HISTORICAL_RISK_MODEL_DIR = MODELS_DIR / "historical_risk_patch_20260416"

# File model utama ConvLSTM historical risk patch.
HISTORICAL_RISK_MODEL_FILE = HISTORICAL_RISK_MODEL_DIR / "best_model_convlstm_historical_risk_patch.keras"

# File laporan training model terbaik.
HISTORICAL_RISK_REPORT_FILE = HISTORICAL_RISK_MODEL_DIR / "training_report_best_1epoch.json"

# File laporan pembanding eksperimen epoch 3.
HISTORICAL_RISK_EPOCH3_REPORT_FILE = HISTORICAL_RISK_MODEL_DIR / "training_report_epoch3_comparison.json"

# Kandidat model lama untuk profile legacy_128.
LEGACY_CONVLSTM_MODEL_CANDIDATES = [
    CONVLSTM_MODEL_FILE,
    CONVLSTM_SAVEDMODEL_DIR,
    CONVLSTM_IPYNB_KERAS_FILE,
    CONVLSTM_IPYNB_H5_FILE,
]

# Kumpulan konfigurasi profile model yang dapat dipakai oleh sistem.
MODEL_PROFILE_CONFIGS = {
    # Profile default: model ConvLSTM patch 160x160 untuk prediksi risiko hotspot.
    "historical_risk_patch_160": {
        # Nama model yang ditampilkan pada dashboard.
        "display_name": "Historical Risk Patch ConvLSTM 160x160",
        # Deskripsi fungsi model pada sistem web.
        "description": "Model patch 160x160 untuk prediksi area risiko hotspot berbasis citra historis.",
        # Kandidat file model yang akan dicoba dimuat oleh InferenceService.
        "model_candidates": [HISTORICAL_RISK_MODEL_FILE],
        # Artefak laporan training utama.
        "training_report": HISTORICAL_RISK_REPORT_FILE,
        # Artefak laporan pembanding training.
        "comparison_report": HISTORICAL_RISK_EPOCH3_REPORT_FILE,
        # Ukuran input patch model.
        "grid_size": 160,
        # Jumlah channel input; 1 berarti mask grayscale hotspot.
        "channels": 1,
        # Jumlah frame historis H-6 sampai H0.
        "time_steps": 7,
        # Mode preprocessing untuk mengambil mask hotspot merah dari citra.
        "preprocess_mode": "hotspot_red_mask",
        # Mode inferensi memotong citra menjadi patch lalu menggabungkan hasil prediksi.
        "inference_mode": "patch_stitch",
        # Ukuran patch yang masuk ke model.
        "patch_size": 160,
        # Pergeseran patch untuk membuat overlap antarpatch.
        "patch_stride": 80,
        # Jumlah patch per batch inferensi.
        "patch_batch_size": 32,
        # Kernel dilasi untuk mempertegas area hotspot pada mask input.
        "input_dilation_kernel": 5,
        # Threshold rekomendasi untuk binary mask hasil prediksi.
        "recommended_threshold": 0.55,
        # Model default tidak mencari model tambahan dari folder Ipynb.
        "discover_ipynb_models": False,
    },
    # Profile alternatif: model lama 128x128 sebagai pembanding/kompatibilitas.
    "legacy_128": {
        # Nama model lama yang ditampilkan pada dashboard.
        "display_name": "Legacy ConvLSTM 128x128",
        # Deskripsi model lama dengan resize langsung.
        "description": "Model lama 128x128 dengan resize langsung seluruh citra.",
        # Kandidat model lama dari folder models dan Ipynb.
        "model_candidates": LEGACY_CONVLSTM_MODEL_CANDIDATES,
        # Tidak memakai laporan training utama pada konfigurasi legacy.
        "training_report": None,
        # Tidak memakai laporan pembanding pada konfigurasi legacy.
        "comparison_report": None,
        # Ukuran input model lama.
        "grid_size": 128,
        # Jumlah channel input model lama.
        "channels": 1,
        # Jumlah frame historis tetap 7.
        "time_steps": 7,
        # Preprocessing tetap memakai mask hotspot merah.
        "preprocess_mode": "hotspot_red_mask",
        # Mode legacy langsung resize seluruh citra.
        "inference_mode": "direct_resize",
        # Tidak memakai patch size.
        "patch_size": None,
        # Tidak memakai patch stride.
        "patch_stride": None,
        # Tidak memakai batch patch.
        "patch_batch_size": None,
        # Kernel dilasi legacy.
        "input_dilation_kernel": 3,
        # Threshold rekomendasi legacy.
        "recommended_threshold": 0.5,
        # Profile legacy boleh mencari model tambahan dari folder Ipynb.
        "discover_ipynb_models": True,
    },
}

# Membaca profile model aktif dari environment variable; default memakai historical_risk_patch_160.
ACTIVE_MODEL_PROFILE = os.getenv("HOTSPOT_MODEL_PROFILE", "historical_risk_patch_160").strip()

# Jika profile tidak tersedia, sistem dikembalikan ke profile default.
if ACTIVE_MODEL_PROFILE not in MODEL_PROFILE_CONFIGS:
    ACTIVE_MODEL_PROFILE = "historical_risk_patch_160"

# Konfigurasi lengkap model aktif berdasarkan profile terpilih.
ACTIVE_MODEL_CONFIG = MODEL_PROFILE_CONFIGS[ACTIVE_MODEL_PROFILE]

# Daftar kandidat model ConvLSTM yang dipakai oleh service inferensi.
CONVLSTM_MODEL_CANDIDATES = ACTIVE_MODEL_CONFIG["model_candidates"]

# Ekstensi file citra yang diizinkan untuk upload.
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Urutan wajib input citra historis untuk prediksi H+1.
REQUIRED_STEMS = ["H-6", "H-5", "H-4", "H-3", "H-2", "H-1", "H0"]

# Batas ukuran file upload dalam MB.
MAX_FILE_MB = 5

# Batas ukuran file upload dalam byte untuk validasi backend.
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Nilai default grid untuk kompatibilitas kode lama.
DEFAULT_GRID_SIZE = 128

# Nilai default channel untuk kompatibilitas kode lama.
DEFAULT_CHANNELS = 1

# Nilai default jumlah time step.
DEFAULT_TIME_STEPS = 7

# Probabilitas minimum yang mulai terlihat pada overlay peta.
OVERLAY_MIN_VISIBLE_PROBABILITY = 0.15

# Lokasi file GeoJSON batas kabupaten/kota Riau.
ADMIN_BOUNDARY_GEOJSON_PATH = PROJECT_ROOT / "data-admin" / "batas_kabupaten_kota_riau_geoboundaries_adm2.geojson"

# Status aktif/nonaktif overlay batas administrasi melalui environment variable.
ADMIN_BOUNDARY_OVERLAY_ENABLED = os.getenv("HOTSPOT_OVERLAY_ADMIN_BOUNDARY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Titik tengah longitude peta Riau untuk proyeksi overlay.
ADMIN_BOUNDARY_CENTER_LON = float(os.getenv("HOTSPOT_MAP_CENTER_LON", "102.1"))

# Titik tengah latitude peta Riau untuk proyeksi overlay.
ADMIN_BOUNDARY_CENTER_LAT = float(os.getenv("HOTSPOT_MAP_CENTER_LAT", "0.8"))

# Zoom peta untuk konversi koordinat GeoJSON ke piksel gambar.
ADMIN_BOUNDARY_ZOOM = float(os.getenv("HOTSPOT_MAP_ZOOM", "8.1"))

# Ketebalan garis batas wilayah administrasi.
ADMIN_BOUNDARY_LINE_WIDTH = int(os.getenv("HOTSPOT_ADMIN_BOUNDARY_LINE_WIDTH", "3"))

# Ketebalan outline batas wilayah agar garis lebih terlihat.
ADMIN_BOUNDARY_OUTLINE_WIDTH = int(os.getenv("HOTSPOT_ADMIN_BOUNDARY_OUTLINE_WIDTH", "5"))

# Status aktif/nonaktif warna isi wilayah administrasi.
ADMIN_REGION_FILL_ENABLED = os.getenv("HOTSPOT_ADMIN_REGION_FILL", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Transparansi warna isi wilayah administrasi.
ADMIN_REGION_FILL_ALPHA = int(os.getenv("HOTSPOT_ADMIN_REGION_FILL_ALPHA", "44"))

# Status aktif/nonaktif label nama wilayah administrasi.
ADMIN_REGION_LABEL_ENABLED = os.getenv("HOTSPOT_ADMIN_REGION_LABEL", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Ukuran font label wilayah administrasi.
ADMIN_REGION_LABEL_FONT_SIZE = int(os.getenv("HOTSPOT_ADMIN_REGION_LABEL_FONT_SIZE", "22"))

# Luas minimum wilayah agar label/nomor wilayah ditampilkan.
ADMIN_REGION_LABEL_MIN_AREA = float(os.getenv("HOTSPOT_ADMIN_REGION_LABEL_MIN_AREA", "650.0"))


# Fungsi untuk memastikan seluruh folder kerja sistem tersedia sebelum aplikasi berjalan.
def initialize_directories() -> None:
    # Daftar folder yang harus ada agar proses upload, inferensi, dan penyimpanan output berjalan.
    for path in [
        STORAGE_DIR,
        INPUTS_DIR,
        OUTPUTS_DIR,
        ASSETS_DIR,
        STATIC_DIR,
        TEMPLATES_DIR,
        MODELS_DIR,
        HISTORICAL_RISK_MODEL_DIR,
        IPYNB_DIR,
    ]:
        # Membuat folder secara rekursif dan tidak error jika folder sudah ada.
        path.mkdir(parents=True, exist_ok=True)