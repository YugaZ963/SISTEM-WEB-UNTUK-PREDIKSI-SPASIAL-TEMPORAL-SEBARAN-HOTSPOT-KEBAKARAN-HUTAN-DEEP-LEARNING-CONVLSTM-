
"""Komentar file skripsi:
Script training utama model historical_risk_patch berbasis dataset citra hotspot PNG.

Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
terutama data understanding, data preparation, modeling, evaluation,
atau deployment sesuai fungsi teknisnya.
"""

from __future__ import annotations

# argparse dipakai agar parameter dataset, patch, threshold, dan output bisa diatur dari command line.
import argparse
# json dipakai untuk menyimpan ringkasan eksperimen, metrik, dan metadata proses.
import json
# math dipakai untuk perhitungan ukuran patch, skor, atau pembulatan metrik numerik.
import math
# random dipakai untuk sampling patch agar variasi contoh hotspot dapat dikontrol.
import random
# re dipakai untuk membaca tanggal dari nama file FIRMS_YYYY-MM-DD.
import re
# dataclass dipakai untuk merepresentasikan satu record citra hotspot secara terstruktur.
from dataclasses import dataclass
# datetime dipakai untuk menjaga urutan kronologis citra dan menghitung target H+1.
from datetime import datetime, timedelta, timezone
# lru_cache dipakai agar mask citra yang sama tidak dibuka dan dihitung ulang berkali-kali.
from functools import lru_cache
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path
from zoneinfo import ZoneInfo

# Flag ini membedakan konfigurasi lokal dan Google Colab saat training model.
RUNNING_IN_COLAB = False

try:
    from google.colab import drive  # type: ignore

    # Flag ini membedakan konfigurasi lokal dan Google Colab saat training model.
    RUNNING_IN_COLAB = True
except Exception:
    drive = None  # type: ignore

# NumPy dipakai untuk mask hotspot, tensor sequence, patch, probability map, dan metrik piksel.
import numpy as np
# TensorFlow dipakai untuk membangun, melatih, memuat, dan menjalankan model ConvLSTM.
import tensorflow as tf
# Pillow Image dipakai untuk membuka citra hotspot PNG/JPG dan menyimpan preview/hasil.
from PIL import Image, ImageFilter
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
from tensorflow.keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Dropout, Input


# Regex ini mengambil tanggal dari nama file dataset FIRMS agar urutan temporal dapat dibentuk.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")
# Ekstensi default ini membatasi file yang dianggap sebagai citra hotspot valid.
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# Mencari root project agar script training tetap bisa berjalan dari Colab, WSL, atau folder lokal.
def discover_project_root() -> Path:
    base_candidates: list[Path] = []
    if "__file__" in globals():
        base_candidates.append(Path(__file__).resolve().parent)
    base_candidates.append(Path.cwd().resolve())

    for base in base_candidates:
        for candidate in [base, *base.parents]:
            if (candidate / "Ipynb").exists():
                # Hasil ini dikembalikan sebagai output fungsi `discover_project_root` untuk tahap berikutnya.
                return candidate
    # Hasil ini dikembalikan sebagai output fungsi `discover_project_root` untuk tahap berikutnya.
    return base_candidates[0]


# Root project dipakai sebagai dasar path dataset, artefak, dan script pendukung.
PROJECT_ROOT = discover_project_root()
# Folder Ipynb menjadi lokasi dataset dan artefak eksperimen model.
IPYNB_DIR = PROJECT_ROOT / "Ipynb" if (PROJECT_ROOT / "Ipynb").exists() else PROJECT_ROOT
# Lokasi dataset citra hotspot historis versi lokal sebelum dikonversi/diolah.
LOCAL_DATASET_DIR = IPYNB_DIR / "Dataset History Fire Hotspot In Riau Province"
# Lokasi dataset PNG yang dipakai pipeline historical risk patch.
LOCAL_PNG_DATASET_DIR = IPYNB_DIR / "Dataset History Fire Hotspot In Riau Province PNG"
# Folder output lokal untuk model dan laporan training.
LOCAL_ARTIFACTS_DIR = IPYNB_DIR / "artifacts-native"
# Kandidat folder dataset saat training dijalankan di Google Colab.
COLAB_DATASET_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/data/dataset_riau/Dataset_History_Fire_Hotspot_In_Riau_Province"
)
COLAB_ALT_DATASET_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/data/Dataset_History_Fire_Hotspot_In_Riau_Province"
)
# Lokasi penyimpanan model ConvLSTM hasil training di Google Drive.
COLAB_OUTPUT_MODEL = Path(
    "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/artifacts-native/best_model_convlstm_historical_risk_patch_1528x773.keras"
)
# Lokasi laporan JSON training di Google Drive.
COLAB_OUTPUT_REPORT = Path(
    "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/artifacts-native/convlstm_historical_risk_patch_1528x773_training_report.json"
)
# Lokasi model hasil training jika script dijalankan di mesin lokal.
LOCAL_OUTPUT_MODEL = LOCAL_ARTIFACTS_DIR / "best_model_convlstm_historical_risk_patch_1528x773.keras"
# Lokasi laporan training jika script dijalankan di mesin lokal.
LOCAL_OUTPUT_REPORT = LOCAL_ARTIFACTS_DIR / "convlstm_historical_risk_patch_1528x773_training_report.json"


# Mount Google Drive hanya saat script berjalan di Colab, karena dataset/model training biasanya berada di Drive.
def maybe_mount_drive() -> None:
    if not RUNNING_IN_COLAB:
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return
    try:
        drive.mount("/content/drive")  # type: ignore[attr-defined]
    except Exception as exc:
        print("Drive mount dilewati:", exc)


# Mengaktifkan memory growth GPU agar TensorFlow tidak langsung mengambil seluruh memori GPU.
def configure_tensorflow_runtime() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print("GPU terdeteksi:", ", ".join(device.name for device in gpus))
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        print("GPU tidak terdeteksi. Training akan berjalan di CPU.")


# Decorator ini membuat record dataset/statistik lebih ringkas tanpa menulis constructor manual.
@dataclass(frozen=True)
# Class ini menyatukan path citra hotspot dan tanggalnya agar urutan temporal tidak tertukar.
class DatasetRecord:
    path: Path
    date: datetime


# Membuat nama artefak model/report dengan timestamp WIB supaya hasil training lama tidak tertimpa.
def build_versioned_path(path: Path, label: str) -> Path:
    try:
        wib_now = datetime.now(ZoneInfo("Asia/Jakarta"))
    except Exception:
        wib_now = datetime.now(timezone(timedelta(hours=7)))
    timestamp = wib_now.strftime("%Y%m%d_%H%M%S_WIB")
    # Hasil ini dikembalikan sebagai output fungsi `build_versioned_path` untuk tahap berikutnya.
    return path.with_name(f"{path.stem}_{label}_{timestamp}{path.suffix}")


# Menentukan apakah model/report training ditimpa atau disimpan sebagai file run baru.
def resolve_output_paths(output_model: Path, output_report: Path, overwrite: bool) -> tuple[Path, Path]:
    if overwrite:
        # Hasil ini dikembalikan sebagai output fungsi `resolve_output_paths` untuk tahap berikutnya.
        return output_model, output_report
    if not output_model.exists() and not output_report.exists():
        # Hasil ini dikembalikan sebagai output fungsi `resolve_output_paths` untuk tahap berikutnya.
        return output_model, output_report

    versioned_model = build_versioned_path(output_model, "run")
    versioned_report = build_versioned_path(output_report, "run")
    print("Artefak lama sudah ada. Menyimpan run baru tanpa menimpa file lama.")
    print(f"Model baru: {versioned_model}")
    print(f"Report baru: {versioned_report}")
    # Hasil ini dikembalikan sebagai output fungsi `resolve_output_paths` untuk tahap berikutnya.
    return versioned_model, versioned_report


# Memilih folder dataset hotspot yang benar dari kandidat lokal, project, dan Google Drive.
def resolve_dataset_dir(dataset_dir_arg: str) -> Path:
    # `direct_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    direct_path = Path(dataset_dir_arg).expanduser()
    candidates = [direct_path]
    variants = [
        LOCAL_DATASET_DIR,
        PROJECT_ROOT / "Dataset History Fire Hotspot In Riau Province",
        COLAB_DATASET_DIR,
        COLAB_ALT_DATASET_DIR,
    ]
    for candidate in variants:
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            # Hasil ini dikembalikan sebagai output fungsi `resolve_dataset_dir` untuk tahap berikutnya.
            return candidate.resolve()

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Dataset tidak ditemukan. Path yang dicek:\n"
        f"{searched}\n\n"
        "Pastikan Google Drive sudah termount dan nama folder di Drive sesuai."
    )


# Membaca semua parameter training seperti dataset, patch, threshold, batch, epoch, dan lokasi output.
def parse_args() -> argparse.Namespace:
    # `default_dataset_dir` adalah nilai bawaan yang dipilih sesuai lingkungan lokal atau Colab.
    default_dataset_dir = COLAB_DATASET_DIR if RUNNING_IN_COLAB else LOCAL_DATASET_DIR
    # `default_output_model` adalah nilai bawaan yang dipilih sesuai lingkungan lokal atau Colab.
    default_output_model = COLAB_OUTPUT_MODEL if RUNNING_IN_COLAB else LOCAL_OUTPUT_MODEL
    # `default_output_report` adalah nilai bawaan yang dipilih sesuai lingkungan lokal atau Colab.
    default_output_report = COLAB_OUTPUT_REPORT if RUNNING_IN_COLAB else LOCAL_OUTPUT_REPORT
    # `default_patch_size` adalah nilai bawaan yang dipilih sesuai lingkungan lokal atau Colab.
    default_patch_size = 192 if RUNNING_IN_COLAB else 160
    # `default_batch_size` adalah nilai bawaan yang dipilih sesuai lingkungan lokal atau Colab.
    default_batch_size = 4 if RUNNING_IN_COLAB else 1
    # `default_epochs` adalah nilai bawaan yang dipilih sesuai lingkungan lokal atau Colab.
    default_epochs = 30 if RUNNING_IN_COLAB else 20

    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description=(
            "Latih ConvLSTM native patch 1528x773 dengan target heatmap risiko "
            "dari dataset hotspot JPG/JPEG/PNG."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(default_dataset_dir),
    )
    parser.add_argument(
        "--output-model",
        default=str(default_output_model),
    )
    parser.add_argument(
        "--output-report",
        default=str(default_output_report),
    )
    # Opsi ini menyimpan lebar asli peta/citra project.
    parser.add_argument("--native-width", type=int, default=1528)
    # Opsi ini menyimpan tinggi asli peta/citra project.
    parser.add_argument("--native-height", type=int, default=773)
    # Opsi ini menentukan lebar patch yang dipotong dari citra besar.
    parser.add_argument("--patch-width", type=int, default=default_patch_size)
    # Opsi ini menentukan tinggi patch yang dipotong dari citra besar.
    parser.add_argument("--patch-height", type=int, default=default_patch_size)
    # Opsi ini menentukan jumlah frame historis; project memakai 7 frame H-6 sampai H0.
    parser.add_argument("--seq-length", type=int, default=7)
    # Opsi ini mengatur jumlah putaran training model.
    parser.add_argument("--epochs", type=int, default=default_epochs)
    # Opsi ini mengatur jumlah patch/sequence yang diproses sekali training.
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    # Opsi ini mengatur porsi sequence untuk training secara kronologis.
    parser.add_argument("--train-ratio", type=float, default=0.70)
    # Opsi ini mengatur porsi sequence untuk validation setelah training.
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # Opsi ini menentukan ambang binary mask pada evaluasi atau visualisasi.
    parser.add_argument("--threshold", type=float, default=0.3)
    # Opsi ini menentukan batas target risk map yang dihitung sebagai positif.
    parser.add_argument("--ground-truth-threshold", type=float, default=0.05)
    # Opsi `--dilation-kernel` menambah parameter eksekusi script.
    parser.add_argument("--dilation-kernel", type=int, default=5)
    parser.add_argument(
        "--input-dilation-kernel",
        type=int,
        default=5,
        help="Dilasi untuk mask input historis. Default 5 agar sama dengan baseline native patch.",
    )
    parser.add_argument(
        "--label-dilation-kernel",
        type=int,
        default=9,
        help="Dilasi target risiko. Nilai lebih besar membentuk area risiko di sekitar hotspot.",
    )
    parser.add_argument(
        "--label-blur-radius",
        type=float,
        default=2.0,
        help="Radius blur untuk membuat target density/heatmap lebih halus. Pakai 0 untuk menonaktifkan.",
    )
    parser.add_argument(
        "--evaluation-buffer-radius",
        type=int,
        default=5,
        help="Radius buffer evaluasi berbasis toleransi spasial pada patch output.",
    )
    # Opsi `--context-kernel` menambah parameter eksekusi script.
    parser.add_argument("--context-kernel", type=int, default=5)
    # Opsi `--max-pos-weight` menambah parameter eksekusi script.
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument(
        "--loss-strategy",
        choices=["wbce_dice", "wbce_dice_context"],
        default="wbce_dice_context",
    )
    parser.add_argument(
        "--threshold-sweep-step",
        type=float,
        default=0.05,
        help="Langkah coarse threshold sweep validation.",
    )
    # Opsi `--feature-stack` menambah parameter eksekusi script.
    parser.add_argument("--feature-stack", choices=["mask", "mask_context3"], default="mask")
    parser.add_argument(
        "--image-extensions",
        default=",".join(DEFAULT_IMAGE_EXTENSIONS),
        help="Daftar ekstensi gambar yang dibaca, pisahkan dengan koma. Contoh: .jpg,.jpeg,.png",
    )
    # Opsi `--train-positive-patches` menambah parameter eksekusi script.
    parser.add_argument("--train-positive-patches", type=int, default=3)
    # Opsi `--train-negative-patches` menambah parameter eksekusi script.
    parser.add_argument("--train-negative-patches", type=int, default=1)
    # Opsi `--eval-positive-patches` menambah parameter eksekusi script.
    parser.add_argument("--eval-positive-patches", type=int, default=1)
    # Opsi `--eval-negative-patches` menambah parameter eksekusi script.
    parser.add_argument("--eval-negative-patches", type=int, default=1)
    # Opsi `--enable-augmentation` menambah parameter eksekusi script.
    parser.add_argument("--enable-augmentation", action="store_true")
    # Opsi `--seed` menambah parameter eksekusi script.
    parser.add_argument("--seed", type=int, default=42)
    # Opsi `--overwrite` menambah parameter eksekusi script.
    parser.add_argument("--overwrite", action="store_true")
    args, _ = parser.parse_known_args()
    # Hasil ini dikembalikan sebagai output fungsi `parse_args` untuk tahap berikutnya.
    return args


# Fungsi `is_notebook_runtime` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def is_notebook_runtime() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        # Hasil ini dikembalikan sebagai output fungsi `is_notebook_runtime` untuk tahap berikutnya.
        return False
    shell = get_ipython()
    if shell is None:
        # Hasil ini dikembalikan sebagai output fungsi `is_notebook_runtime` untuk tahap berikutnya.
        return False
    # Hasil ini dikembalikan sebagai output fungsi `is_notebook_runtime` untuk tahap berikutnya.
    return shell.__class__.__name__ in {"ZMQInteractiveShell", "Shell"}


# Fungsi `feature_stack_channels` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def feature_stack_channels(feature_stack: str) -> int:
    # Hasil ini dikembalikan sebagai output fungsi `feature_stack_channels` untuk tahap berikutnya.
    return 3 if feature_stack == "mask_context3" else 1


# Menormalisasi daftar ekstensi citra agar dataset PNG/JPG bisa dipindai konsisten.
def parse_image_extensions(value: str) -> tuple[str, ...]:
    extensions: list[str] = []
    for item in value.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in extensions:
            extensions.append(ext)
    # Hasil ini dikembalikan sebagai output fungsi `parse_image_extensions` untuk tahap berikutnya.
    return tuple(extensions) if extensions else DEFAULT_IMAGE_EXTENSIONS


# Membaca file citra hotspot dari folder dataset dan mengurutkannya berdasarkan tanggal pada nama file.
def load_records(dataset_dir: Path, image_extensions: tuple[str, ...]) -> list[DatasetRecord]:
    # Daftar record citra hotspot yang sudah terbaca dari folder dataset.
    records: list[DatasetRecord] = []
    allowed_extensions = set(image_extensions)
    for path in sorted(item for item in dataset_dir.iterdir() if item.is_file()):
        # File dilewati jika ekstensinya bukan citra yang diizinkan untuk dataset hotspot.
        if path.suffix.lower() not in allowed_extensions:
            continue
        # Nama file diperiksa agar tanggal FIRMS bisa dipakai sebagai urutan waktu.
        match = DATE_PATTERN.search(path.name)
        if not match:
            continue
        records.append(DatasetRecord(path=path, date=datetime.strptime(match.group(1), "%Y-%m-%d")))
    # Hasil ini dikembalikan sebagai output fungsi `load_records` untuk tahap berikutnya.
    return sorted(records, key=lambda item: item.date)


# Memastikan setiap citra dataset benar-benar bisa dibuka sebelum dipakai membentuk sequence.
def validate_records(records: list[DatasetRecord]) -> tuple[list[DatasetRecord], list[dict[str, str]]]:
    # Daftar citra yang lolos validasi file gambar dan bisa dipakai membentuk sequence.
    valid_records: list[DatasetRecord] = []
    # Daftar citra yang dilewati beserta alasan, misalnya file rusak atau nama tidak bertanggal.
    skipped_records: list[dict[str, str]] = []
    # Loop ini memproses setiap citra hotspot yang telah terurut kronologis.
    for record in records:
        try:
            if not record.path.exists():
                raise FileNotFoundError(f"File tidak ditemukan: {record.path}")
            # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
            with Image.open(record.path) as image:
                # Verifikasi ini memastikan file gambar tidak rusak sebelum masuk pipeline.
                image.verify()
            valid_records.append(record)
        except Exception as exc:
            skipped_records.append({"path": str(record.path), "reason": str(exc)})
    # Hasil ini dikembalikan sebagai output fungsi `validate_records` untuk tahap berikutnya.
    return valid_records, skipped_records


# Memastikan ukuran kernel dilation/blur bersifat ganjil karena filter gambar membutuhkan pusat kernel.
def _normalize_kernel(size: int) -> int:
    size = max(1, int(size))
    # Hasil ini dikembalikan sebagai output fungsi `_normalize_kernel` untuk tahap berikutnya.
    return size if size % 2 == 1 else size + 1

# Cache ini mempercepat proses karena citra/mask yang sama sering dipakai berulang saat patch sampling.
@lru_cache(maxsize=96)
# Mengubah citra hotspot menjadi mask input satu channel berdasarkan piksel merah hotspot.
def load_native_mask(
    path_str: str,
    native_width: int,
    native_height: int,
    dilation_kernel: int,
) -> np.ndarray:
    dilation_kernel = _normalize_kernel(dilation_kernel)
    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    with Image.open(path_str) as image:
        rgb = image.convert("RGB")
        # Representasi HSV memudahkan pemisahan warna merah hotspot dibanding RGB langsung.
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)

    # Channel Hue dipakai untuk menentukan rentang warna merah hotspot.
    h = hsv[:, :, 0]
    # Channel Saturation memastikan piksel yang dipilih benar-benar berwarna kuat, bukan abu-abu.
    s = hsv[:, :, 1]
    # Channel Value memastikan piksel hotspot cukup terang untuk dihitung sebagai sinyal.
    v = hsv[:, :, 2]
    # Rentang merah bawah pada HSV untuk menangkap piksel hotspot.
    red_low = (h <= 14) & (s >= 70) & (v >= 50)
    # Rentang merah atas pada HSV untuk menangkap wrap-around warna merah.
    red_high = (h >= 242) & (s >= 70) & (v >= 50)
    # Mask biner hotspot: piksel merah menjadi positif dan background menjadi nol.
    mask = ((red_low | red_high).astype(np.uint8)) * 255

    # Gambar mask dipakai agar dilation/resize bisa memakai operasi Pillow.
    mask_image = Image.fromarray(mask)
    if dilation_kernel > 1:
        # Gambar mask dipakai agar dilation/resize bisa memakai operasi Pillow.
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=dilation_kernel))
    if mask_image.size != (native_width, native_height):
        # Gambar mask dipakai agar dilation/resize bisa memakai operasi Pillow.
        mask_image = mask_image.resize((native_width, native_height), Image.BILINEAR)

    # Array 0-1 yang menjadi input model atau target risk map.
    density = np.asarray(mask_image, dtype=np.float32) / 255.0
    # Nilai dikunci ke rentang 0-1 agar cocok sebagai mask/probability map.
    return np.clip(density, 0.0, 1.0)


# Cache ini mempercepat proses karena citra/mask yang sama sering dipakai berulang saat patch sampling.
@lru_cache(maxsize=96)
# Membentuk target risk map H+1 dari hotspot merah, dilation label, dan Gaussian blur.
def load_native_risk_map(
    path_str: str,
    native_width: int,
    native_height: int,
    label_dilation_kernel: int,
    label_blur_radius: float,
) -> np.ndarray:
    """Build a softer target map so hotspot labels represent nearby risk areas."""
    label_dilation_kernel = _normalize_kernel(label_dilation_kernel)
    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    with Image.open(path_str) as image:
        rgb = image.convert("RGB")
        # Representasi HSV memudahkan pemisahan warna merah hotspot dibanding RGB langsung.
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)

    # Channel Hue dipakai untuk menentukan rentang warna merah hotspot.
    h = hsv[:, :, 0]
    # Channel Saturation memastikan piksel yang dipilih benar-benar berwarna kuat, bukan abu-abu.
    s = hsv[:, :, 1]
    # Channel Value memastikan piksel hotspot cukup terang untuk dihitung sebagai sinyal.
    v = hsv[:, :, 2]
    # Rentang merah bawah pada HSV untuk menangkap piksel hotspot.
    red_low = (h <= 14) & (s >= 70) & (v >= 50)
    # Rentang merah atas pada HSV untuk menangkap wrap-around warna merah.
    red_high = (h >= 242) & (s >= 70) & (v >= 50)
    # Mask biner hotspot: piksel merah menjadi positif dan background menjadi nol.
    mask = ((red_low | red_high).astype(np.uint8)) * 255

    # Gambar target risk map yang akan diperluas dan dihaluskan dari mask hotspot H+1.
    risk_image = Image.fromarray(mask)
    if label_dilation_kernel > 1:
        # Gambar target risk map yang akan diperluas dan dihaluskan dari mask hotspot H+1.
        risk_image = risk_image.filter(ImageFilter.MaxFilter(size=label_dilation_kernel))
    if label_blur_radius > 0:
        # Gambar target risk map yang akan diperluas dan dihaluskan dari mask hotspot H+1.
        risk_image = risk_image.filter(ImageFilter.GaussianBlur(radius=float(label_blur_radius)))
    if risk_image.size != (native_width, native_height):
        # Gambar target risk map yang akan diperluas dan dihaluskan dari mask hotspot H+1.
        risk_image = risk_image.resize((native_width, native_height), Image.BILINEAR)

    # Array 0-1 yang menjadi input model atau target risk map.
    density = np.asarray(risk_image, dtype=np.float32) / 255.0
    # Nilai dikunci ke rentang 0-1 agar cocok sebagai mask/probability map.
    return np.clip(density, 0.0, 1.0)


# Fungsi `build_feature_map` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def build_feature_map(mask: np.ndarray, feature_stack: str, context_kernel: int) -> np.ndarray:
    if feature_stack == "mask_context3":
        context_kernel = _normalize_kernel(context_kernel)
        context_image = Image.fromarray(np.uint8(mask * 255.0))
        if context_kernel > 1:
            context_image = context_image.filter(ImageFilter.BoxBlur(radius=max(1, context_kernel // 2)))
        context_map = np.asarray(context_image, dtype=np.float32) / 255.0
        hotspot_core = (mask >= 0.25).astype(np.float32)
        # Nilai dikunci ke rentang 0-1 agar cocok sebagai mask/probability map.
        return np.stack([mask, np.clip(context_map, 0.0, 1.0), hotspot_core], axis=-1).astype(np.float32)
    # Hasil ini dikembalikan sebagai output fungsi `build_feature_map` untuk tahap berikutnya.
    return np.expand_dims(mask.astype(np.float32), axis=-1)


# Membentuk indeks awal sliding window: tujuh citra historis sebagai input dan frame berikutnya sebagai target H+1.
def build_sample_starts(record_count: int, seq_length: int) -> list[int]:
    # Hasil ini dikembalikan sebagai output fungsi `build_sample_starts` untuk tahap berikutnya.
    return list(range(record_count - seq_length))


# Membagi sequence secara kronologis menjadi train, validation, dan test agar tidak ada kebocoran waktu.
def split_sample_starts(sample_starts: list[int], train_ratio: float, val_ratio: float) -> tuple[list[int], list[int], list[int]]:
    sample_count = len(sample_starts)
    train_end = max(1, int(sample_count * train_ratio))
    val_end = max(train_end + 1, int(sample_count * (train_ratio + val_ratio)))
    val_end = min(val_end, sample_count - 1)
    train = sample_starts[:train_end]
    val = sample_starts[train_end:val_end]
    test = sample_starts[val_end:]
    if not val or not test:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Split train/val/test gagal. Periksa rasio dataset.")
    # Hasil ini dikembalikan sebagai output fungsi `split_sample_starts` untuk tahap berikutnya.
    return train, val, test


# Memilih pusat patch positif dan negatif agar training melihat hotspot dan background secara seimbang.
def sample_patch_centers(
    target_mask: np.ndarray,
    positive_patch_count: int,
    negative_patch_count: int,
    ground_truth_threshold: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    # Mask boolean yang menandai area target di atas threshold ground truth.
    binary = target_mask >= ground_truth_threshold
    # Koordinat piksel/area yang mengandung target hotspot untuk patch positif.
    positive_coords = np.argwhere(binary)
    # Daftar pusat patch yang akan dipotong dari citra besar.
    centers: list[tuple[int, int]] = []

    if len(positive_coords) > 0 and positive_patch_count > 0:
        replace = len(positive_coords) < positive_patch_count
        indices = rng.choice(len(positive_coords), size=positive_patch_count, replace=replace)
        for idx in np.atleast_1d(indices):
            cy, cx = positive_coords[int(idx)]
            centers.append((int(cy), int(cx)))

    neg_needed = negative_patch_count
    attempts = max(neg_needed * 40, 40)
    while neg_needed > 0 and attempts > 0:
        cy = int(rng.integers(0, target_mask.shape[0]))
        cx = int(rng.integers(0, target_mask.shape[1]))
        if not binary[cy, cx]:
            centers.append((cy, cx))
            neg_needed -= 1
        attempts -= 1

    if not centers:
        centers.append((target_mask.shape[0] // 2, target_mask.shape[1] // 2))
    # Hasil ini dikembalikan sebagai output fungsi `sample_patch_centers` untuk tahap berikutnya.
    return centers


# Mengambil patch 160x160 dari mask/risk map dan memberi padding jika pusat patch dekat tepi citra.
def extract_patch(array: np.ndarray, cy: int, cx: int, patch_height: int, patch_width: int) -> np.ndarray:
    if array.ndim == 2:
        array = array[..., None]

    height, width = array.shape[:2]
    half_h = patch_height // 2
    half_w = patch_width // 2
    top = cy - half_h
    left = cx - half_w
    bottom = top + patch_height
    right = left + patch_width

    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - height)
    pad_right = max(0, right - width)

    if pad_top or pad_left or pad_bottom or pad_right:
        array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
        top += pad_top
        left += pad_left
        bottom = top + patch_height
        right = left + patch_width

    # Hasil ini dikembalikan sebagai output fungsi `extract_patch` untuk tahap berikutnya.
    return array[top:bottom, left:right, :].astype(np.float32)


# Menghitung bobot kelas positif agar piksel hotspot yang langka tidak diabaikan model.
def estimate_positive_weight(
    records: list[DatasetRecord],
    sample_starts: list[int],
    seq_length: int,
    native_width: int,
    native_height: int,
    label_dilation_kernel: int,
    label_blur_radius: float,
    max_pos_weight: float,
) -> tuple[float, float]:
    positive_sum = 0.0
    pixel_count = 0
    # Loop ini memproses setiap sequence/sliding window yang menjadi sampel model.
    for start in sample_starts:
        # Mask biner hotspot: piksel merah menjadi positif dan background menjadi nol.
        mask = load_native_risk_map(
            str(records[start + seq_length].path),
            native_width,
            native_height,
            label_dilation_kernel,
            label_blur_radius,
        )
        positive_sum += float(mask.sum())
        pixel_count += mask.size
    # Proporsi piksel hotspot/risk map positif; nilainya kecil karena class imbalance ekstrem.
    positive_ratio = max(positive_sum / max(pixel_count, 1), 1e-6)
    # Bobot kelas positif agar loss lebih memperhatikan piksel hotspot yang langka.
    pos_weight = min((1.0 - positive_ratio) / positive_ratio, max_pos_weight)
    # Hasil ini dikembalikan sebagai output fungsi `estimate_positive_weight` untuk tahap berikutnya.
    return float(max(pos_weight, 1.0)), float(positive_ratio)


# Fungsi `_compute_weighted_bce` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _compute_weighted_bce(y_true: tf.Tensor, y_pred: tf.Tensor, pos_weight_tensor: tf.Tensor) -> tf.Tensor:
    # Hasil ini dikembalikan sebagai output fungsi `_compute_weighted_bce` untuk tahap berikutnya.
    return tf.reduce_mean(
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        -(
            (pos_weight_tensor * y_true * tf.math.log(y_pred))
            + ((1.0 - y_true) * tf.math.log(1.0 - y_pred))
        )
    )


# Fungsi `_compute_dice_loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _compute_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    # Hasil ini dikembalikan sebagai output fungsi `_compute_dice_loss` untuk tahap berikutnya.
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))


# Fungsi `_compute_context_bce` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _compute_context_bce(y_true: tf.Tensor, y_pred: tf.Tensor, context_kernel: int) -> tf.Tensor:
    pooled_context = tf.nn.max_pool2d(y_true, ksize=context_kernel, strides=1, padding="SAME")
    context_weight = 1.0 + (2.0 * pooled_context)
    context_bce_map = -(
        (y_true * tf.math.log(y_pred))
        + ((1.0 - y_true) * tf.math.log(1.0 - y_pred))
    )
    # Hasil ini dikembalikan sebagai output fungsi `_compute_context_bce` untuk tahap berikutnya.
    return tf.reduce_mean(context_weight * context_bce_map)


# Fungsi `make_weighted_bce_dice_loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def make_weighted_bce_dice_loss(pos_weight: float):
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Fungsi `loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        weighted_bce = _compute_weighted_bce(y_true, y_pred, pos_weight_tensor)
        dice_loss = _compute_dice_loss(y_true, y_pred)
        # Hasil ini dikembalikan sebagai output fungsi `loss` untuk tahap berikutnya.
        return (0.7 * weighted_bce) + (0.3 * dice_loss)

    # Hasil ini dikembalikan sebagai output fungsi `loss` untuk tahap berikutnya.
    return loss


# Fungsi `make_weighted_bce_dice_context_loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def make_weighted_bce_dice_context_loss(pos_weight: float, context_kernel: int):
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    context_kernel = _normalize_kernel(context_kernel)
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Fungsi `loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        weighted_bce = _compute_weighted_bce(y_true, y_pred, pos_weight_tensor)
        dice_loss = _compute_dice_loss(y_true, y_pred)
        context_bce = _compute_context_bce(y_true, y_pred, context_kernel)
        base_loss = (0.7 * weighted_bce) + (0.3 * dice_loss)
        # Hasil ini dikembalikan sebagai output fungsi `loss` untuk tahap berikutnya.
        return (0.75 * base_loss) + (0.25 * context_bce)

    # Hasil ini dikembalikan sebagai output fungsi `loss` untuk tahap berikutnya.
    return loss


# Fungsi `build_loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def build_loss(
    loss_strategy: str,
    pos_weight: float,
    context_kernel: int,
):
    if loss_strategy == "wbce_dice_context":
        # Hasil ini dikembalikan sebagai output fungsi `build_loss` untuk tahap berikutnya.
        return make_weighted_bce_dice_context_loss(pos_weight, context_kernel)
    # Hasil ini dikembalikan sebagai output fungsi `build_loss` untuk tahap berikutnya.
    return make_weighted_bce_dice_loss(pos_weight)


# Menyusun arsitektur ConvLSTM yang menerima sequence 7 frame dan menghasilkan peta probabilitas H+1.
def build_model(
    seq_length: int,
    patch_height: int,
    patch_width: int,
    channels: int,
    pos_weight: float,
    loss_strategy: str,
    context_kernel: int,
# Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
) -> tf.keras.Model:
    model = Sequential(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            Input(shape=(seq_length, patch_height, patch_width, channels)),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same"),
        ]
    )
    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    model.compile(
        # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=build_loss(loss_strategy, pos_weight, context_kernel),
        jit_compile=False,
        # Nilai precision, recall, F1, IoU, dan metrik lain hasil evaluasi.
        metrics=[
            # `tf.keras.metrics.BinaryAccuracy(name` menyimpan ukuran evaluasi performa prediksi hotspot.
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            # `tf.keras.metrics.Precision(name` menyimpan ukuran evaluasi performa prediksi hotspot.
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            # `tf.keras.metrics.Recall(name` menyimpan ukuran evaluasi performa prediksi hotspot.
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
            # `tf.keras.metrics.AUC(name` menyimpan ukuran evaluasi performa prediksi hotspot.
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    # Hasil ini dikembalikan sebagai output fungsi `build_model` untuk tahap berikutnya.
    return model


# Membuat wadah bernama `NativePatchSequence` untuk menyimpan data atau aturan kerja.
# Class `NativePatchSequence` menyimpan struktur data khusus untuk tahap project ini.
class NativePatchSequence(tf.keras.utils.Sequence):
    # Fungsi `__init__` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def __init__(
        self,
        records: list[DatasetRecord],
        entries: list[tuple[int, int, int]],
        *,
        seq_length: int,
        native_width: int,
        native_height: int,
        patch_width: int,
        patch_height: int,
        input_dilation_kernel: int,
        label_dilation_kernel: int,
        label_blur_radius: float,
        feature_stack: str,
        context_kernel: int,
        batch_size: int,
        shuffle: bool,
        enable_augmentation: bool,
        seed: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Daftar record citra hotspot yang sudah terbaca dari folder dataset.
        self.records = records
        # Daftar metadata patch yang menghubungkan sample, pusat patch, dan status positif/negatif.
        self.entries = list(entries)
        self.seq_length = seq_length
        self.native_width = native_width
        self.native_height = native_height
        # `patch_width` berkaitan dengan potongan citra 160x160 yang dipakai model.
        self.patch_width = patch_width
        # `patch_height` berkaitan dengan potongan citra 160x160 yang dipakai model.
        self.patch_height = patch_height
        self.input_dilation_kernel = input_dilation_kernel
        self.label_dilation_kernel = label_dilation_kernel
        self.label_blur_radius = label_blur_radius
        self.feature_stack = feature_stack
        self.context_kernel = context_kernel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.enable_augmentation = enable_augmentation
        self.channels = feature_stack_channels(feature_stack)
        self._rng = random.Random(seed)
        self.on_epoch_end()

    # Fungsi `__len__` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def __len__(self) -> int:
        # Hasil ini dikembalikan sebagai output fungsi `__len__` untuk tahap berikutnya.
        return math.ceil(len(self.entries) / self.batch_size)

    # Fungsi `__getitem__` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_entries = self.entries[index * self.batch_size : (index + 1) * self.batch_size]
        x_batch = np.zeros(
            (len(batch_entries), self.seq_length, self.patch_height, self.patch_width, self.channels),
            dtype=np.float32,
        )
        y_batch = np.zeros((len(batch_entries), self.patch_height, self.patch_width, 1), dtype=np.float32)
        sample_cache: dict[int, tuple[list[np.ndarray], np.ndarray]] = {}

        for item_index, (start, cy, cx) in enumerate(batch_entries):
            if start not in sample_cache:
                sequence_features: list[np.ndarray] = []
                for offset in range(self.seq_length):
                    # Mask biner hotspot: piksel merah menjadi positif dan background menjadi nol.
                    mask = load_native_mask(
                        str(self.records[start + offset].path),
                        self.native_width,
                        self.native_height,
                        self.input_dilation_kernel,
                    )
                    sequence_features.append(build_feature_map(mask, self.feature_stack, self.context_kernel))

                # `target_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
                target_mask = load_native_risk_map(
                    str(self.records[start + self.seq_length].path),
                    self.native_width,
                    self.native_height,
                    self.label_dilation_kernel,
                    self.label_blur_radius,
                )
                sample_cache[start] = (sequence_features, target_mask[..., None].astype(np.float32))

            sequence_features, target_mask = sample_cache[start]
            # `x_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
            x_patch = np.stack(
                [extract_patch(feature, cy, cx, self.patch_height, self.patch_width) for feature in sequence_features],
                axis=0,
            )
            # `y_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
            y_patch = extract_patch(target_mask, cy, cx, self.patch_height, self.patch_width)

            if self.enable_augmentation:
                if self._rng.random() < 0.5:
                    # `x_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
                    x_patch = np.flip(x_patch, axis=2)
                    # `y_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
                    y_patch = np.flip(y_patch, axis=1)
                if self._rng.random() < 0.5:
                    # `x_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
                    x_patch = np.flip(x_patch, axis=1)
                    # `y_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
                    y_patch = np.flip(y_patch, axis=0)

            x_batch[item_index] = x_patch
            y_batch[item_index] = y_patch
        # Hasil ini dikembalikan sebagai output fungsi `__getitem__` untuk tahap berikutnya.
        return x_batch, y_batch

    # Fungsi `on_epoch_end` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self.entries)


# Fungsi `collect_targets` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def collect_targets(sequence: NativePatchSequence) -> np.ndarray:
    targets: list[np.ndarray] = []
    for batch_index in range(len(sequence)):
        _, y_batch = sequence[batch_index]
        targets.append(y_batch)
    # Hasil ini dikembalikan sebagai output fungsi `collect_targets` untuk tahap berikutnya.
    return np.concatenate(targets, axis=0)


# Fungsi `compute_metrics` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, ground_truth_threshold: float) -> dict[str, float]:
    truth = y_true >= ground_truth_threshold
    pred = y_pred >= threshold
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())

    # Fungsi `safe_div` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def safe_div(a: float, b: float) -> float:
        # Hasil ini dikembalikan sebagai output fungsi `safe_div` untuk tahap berikutnya.
        return 0.0 if b == 0 else float(a / b)

    # `precision` menyimpan ukuran evaluasi performa prediksi hotspot.
    precision = safe_div(tp, tp + fp)
    # `recall` menyimpan ukuran evaluasi performa prediksi hotspot.
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    # `iou` menyimpan ukuran evaluasi performa prediksi hotspot.
    iou = safe_div(tp, tp + fp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    # Hasil ini dikembalikan sebagai output fungsi `safe_div` untuk tahap berikutnya.
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "accuracy": accuracy,
    }


# Fungsi `_dilate_binary_array` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _dilate_binary_array(mask: np.ndarray, buffer_radius: int) -> np.ndarray:
    if buffer_radius <= 0:
        # Hasil ini dikembalikan sebagai output fungsi `_dilate_binary_array` untuk tahap berikutnya.
        return mask.astype(bool)

    original_shape = mask.shape
    squeezed = np.squeeze(mask.astype(np.uint8), axis=-1) if mask.ndim == 4 and mask.shape[-1] == 1 else mask.astype(np.uint8)
    if squeezed.ndim == 2:
        flat = squeezed.reshape(1, squeezed.shape[0], squeezed.shape[1])
    elif squeezed.ndim == 3:
        flat = squeezed.reshape(-1, squeezed.shape[-2], squeezed.shape[-1])
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError(f"Shape mask tidak didukung untuk buffer evaluation: {mask.shape}")

    kernel_size = _normalize_kernel((int(buffer_radius) * 2) + 1)
    dilated: list[np.ndarray] = []
    for item in flat:
        image = Image.fromarray((item > 0).astype(np.uint8) * 255)
        # MaxFilter memperluas area positif sehingga hotspot kecil tidak hilang saat dipatch.
        image = image.filter(ImageFilter.MaxFilter(size=kernel_size))
        dilated.append(np.asarray(image, dtype=np.uint8) > 0)

    stacked = np.stack(dilated, axis=0)
    if squeezed.ndim == 2:
        result = stacked[0]
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        result = stacked.reshape(squeezed.shape)

    if len(original_shape) == 4 and original_shape[-1] == 1:
        result = np.expand_dims(result, axis=-1)
    # Hasil ini dikembalikan sebagai output fungsi `_dilate_binary_array` untuk tahap berikutnya.
    return result.astype(bool)


# Fungsi `compute_buffered_metrics` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def compute_buffered_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    ground_truth_threshold: float,
    buffer_radius: int,
) -> dict[str, float]:
    truth = y_true >= ground_truth_threshold
    pred = y_pred >= threshold
    truth = _dilate_binary_array(truth, buffer_radius)
    pred = _dilate_binary_array(pred, buffer_radius)

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())

    # Fungsi `safe_div` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def safe_div(a: float, b: float) -> float:
        # Hasil ini dikembalikan sebagai output fungsi `safe_div` untuk tahap berikutnya.
        return 0.0 if b == 0 else float(a / b)

    # `precision` menyimpan ukuran evaluasi performa prediksi hotspot.
    precision = safe_div(tp, tp + fp)
    # `recall` menyimpan ukuran evaluasi performa prediksi hotspot.
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    # `iou` menyimpan ukuran evaluasi performa prediksi hotspot.
    iou = safe_div(tp, tp + fp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    # Hasil ini dikembalikan sebagai output fungsi `safe_div` untuk tahap berikutnya.
    return {
        "buffer_radius": int(buffer_radius),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "accuracy": accuracy,
    }


# Fungsi `_threshold_grid` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _threshold_grid(step: float, *, start: float = 0.05, stop: float = 0.95) -> list[float]:
    step = max(float(step), 0.001)
    start = max(0.01, float(start))
    stop = min(0.99, float(stop))
    if stop < start:
        start, stop = stop, start
    values = np.arange(start, stop + (step * 0.5), step)
    # Nilai dikunci ke rentang 0-1 agar cocok sebagai mask/probability map.
    return sorted({round(float(np.clip(value, 0.01, 0.99)), 4) for value in values})


# Fungsi `_metrics_rank` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _metrics_rank(metrics: dict[str, float]) -> tuple[float, float, float, float, int]:
    # Hasil ini dikembalikan sebagai output fungsi `_metrics_rank` untuk tahap berikutnya.
    return (
        metrics["f1_score"],
        metrics["iou"],
        metrics["precision"],
        metrics["recall"],
        -metrics["fp"],
    )


# Fungsi `sweep_thresholds` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def sweep_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ground_truth_threshold: float,
    sweep_step: float,
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    evaluated: dict[float, dict[str, float]] = {}
    # `best_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
    best_threshold = 0.5
    # `best_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    best_metrics: dict[str, float] | None = None

    # Fungsi `evaluate_thresholds` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def evaluate_thresholds(thresholds: list[float]) -> None:
        nonlocal best_threshold, best_metrics
        # Loop ini mencoba beberapa threshold untuk mencari ambang evaluasi terbaik.
        for threshold in thresholds:
            if threshold not in evaluated:
                # `evaluated[threshold]` berkaitan dengan ambang untuk membedakan area risiko dan background.
                evaluated[threshold] = compute_metrics(y_true, y_pred, threshold, ground_truth_threshold)
            # Nilai precision, recall, F1, IoU, dan metrik lain hasil evaluasi.
            metrics = evaluated[threshold]
            if best_metrics is None or _metrics_rank(metrics) > _metrics_rank(best_metrics):
                # `best_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
                best_threshold = threshold
                # `best_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
                best_metrics = metrics

    evaluate_thresholds(_threshold_grid(sweep_step))

    assert best_metrics is not None
    results = [{"threshold": threshold, **evaluated[threshold]} for threshold in sorted(evaluated)]
    # Hasil ini dikembalikan sebagai output fungsi `evaluate_thresholds` untuk tahap berikutnya.
    return best_threshold, results, best_metrics


# Membuat wadah bernama `ValidationSweepCallback` untuk menyimpan data atau aturan kerja.
# Class `ValidationSweepCallback` menyimpan struktur data khusus untuk tahap project ini.
class ValidationSweepCallback(tf.keras.callbacks.Callback):
    # Fungsi `__init__` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def __init__(
        self,
        val_sequence: NativePatchSequence,
        y_val: np.ndarray,
        ground_truth_threshold: float,
        sweep_step: float,
    ) -> None:
        super().__init__()
        self.val_sequence = val_sequence
        self.y_val = y_val
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        self.ground_truth_threshold = ground_truth_threshold
        self.sweep_step = sweep_step

    # Fungsi `on_epoch_end` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        # Model menghasilkan probability map risiko hotspot dari input sequence.
        predictions = self.model.predict(self.val_sequence, verbose=0)
        best_threshold, _, best_metrics = sweep_thresholds(
            self.y_val,
            predictions,
            self.ground_truth_threshold,
            self.sweep_step,
        )
        # `logs["val_best_threshold"]` berkaitan dengan ambang untuk membedakan area risiko dan background.
        logs["val_best_threshold"] = float(best_threshold)
        logs["val_best_precision"] = float(best_metrics["precision"])
        logs["val_best_recall"] = float(best_metrics["recall"])
        logs["val_best_f1"] = float(best_metrics["f1_score"])
        logs["val_best_iou"] = float(best_metrics["iou"])
        logs["val_model_score"] = float(self._compute_model_score(best_metrics))
        print(
            f"\nEpoch {epoch + 1}: val_best_threshold={best_threshold:.2f} "
            f"val_best_f1={best_metrics['f1_score']:.4f} "
            f"val_best_iou={best_metrics['iou']:.4f} "
            f"val_best_recall={best_metrics['recall']:.4f}"
        )

    @staticmethod
    # Fungsi `_compute_model_score` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def _compute_model_score(metrics: dict[str, float]) -> float:
        if metrics["precision"] < 0.02 or metrics["f1_score"] < 0.02:
            # Hasil ini dikembalikan sebagai output fungsi `_compute_model_score` untuk tahap berikutnya.
            return 0.0
        # Hasil ini dikembalikan sebagai output fungsi `_compute_model_score` untuk tahap berikutnya.
        return float(
            (0.60 * metrics["f1_score"])
            + (0.20 * metrics["iou"])
            + (0.10 * metrics["precision"])
            + (0.10 * metrics["recall"])
        )


# Membuat daftar patch training/validasi/test yang menghubungkan sample, pusat patch, dan label positif.
def build_patch_entries(
    records: list[DatasetRecord],
    sample_starts: list[int],
    *,
    seq_length: int,
    native_width: int,
    native_height: int,
    label_dilation_kernel: int,
    label_blur_radius: float,
    ground_truth_threshold: float,
    positive_patches: int,
    negative_patches: int,
    seed: int,
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    rng = np.random.default_rng(seed)
    # Daftar metadata patch yang menghubungkan sample, pusat patch, dan status positif/negatif.
    entries: list[tuple[int, int, int]] = []
    positive_samples = 0
    negative_samples = 0

    # Loop ini memproses setiap sequence/sliding window yang menjadi sampel model.
    for start in sample_starts:
        # `target_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
        target_mask = load_native_risk_map(
            str(records[start + seq_length].path),
            native_width,
            native_height,
            label_dilation_kernel,
            label_blur_radius,
        )
        has_positive = bool(np.any(target_mask >= ground_truth_threshold))
        if has_positive:
            positive_samples += 1
        # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
        else:
            negative_samples += 1

        # Daftar pusat patch yang akan dipotong dari citra besar.
        centers = sample_patch_centers(
            target_mask,
            # `positive_patch_count` berkaitan dengan potongan citra 160x160 yang dipakai model.
            positive_patch_count=positive_patches if has_positive else 0,
            # `negative_patch_count` berkaitan dengan potongan citra 160x160 yang dipakai model.
            negative_patch_count=negative_patches,
            # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
            ground_truth_threshold=ground_truth_threshold,
            rng=rng,
        )
        entries.extend((start, cy, cx) for cy, cx in centers)

    stats = {
        "sample_count": len(sample_starts),
        "positive_sample_count": positive_samples,
        "negative_sample_count": negative_samples,
        "patch_entry_count": len(entries),
        "positive_patches_per_sample": positive_patches,
        "negative_patches_per_sample": negative_patches,
    }
    # Hasil ini dikembalikan sebagai output fungsi `build_patch_entries` untuk tahap berikutnya.
    return entries, stats


# Fungsi `run_training` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def run_training(args: argparse.Namespace, output_model: Path, output_report: Path) -> dict:
    # `dataset_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    image_extensions = parse_image_extensions(args.image_extensions)
    # Daftar record citra hotspot yang sudah terbaca dari folder dataset.
    records = load_records(dataset_dir, image_extensions)
    # Pengecekan ini memastikan jumlah citra cukup untuk membentuk sequence tujuh hari dan target H+1.
    if len(records) <= args.seq_length:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Jumlah data tidak cukup untuk membentuk sequence.")

    records, skipped_records = validate_records(records)
    if not records:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Tidak ada file gambar valid di dataset. Periksa --dataset-dir dan --image-extensions.")
    if skipped_records:
        print(f"File dataset yang dilewati: {len(skipped_records)}")
        # Loop ini memproses setiap citra hotspot yang telah terurut kronologis.
        for item in skipped_records[:10]:
            print(f"- skip {item['path']} | alasan: {item['reason']}")
    # Pengecekan ini memastikan jumlah citra cukup untuk membentuk sequence tujuh hari dan target H+1.
    if len(records) <= args.seq_length:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Jumlah data valid tidak cukup untuk membentuk sequence.")

    # Indeks awal tiap sequence input H-6 sampai H0 untuk memprediksi target H+1.
    sample_starts = build_sample_starts(len(records), args.seq_length)
    train_starts, val_starts, test_starts = split_sample_starts(sample_starts, args.train_ratio, args.val_ratio)
    print(f"[historical_risk_patch] Dataset ditemukan: {len(records)} frame")
    print(f"[historical_risk_patch] Rentang data: {records[0].date.date()} s.d. {records[-1].date.date()}")
    print(f"[historical_risk_patch] Train samples: {len(train_starts)}")
    print(f"[historical_risk_patch] Val samples: {len(val_starts)}")
    print(f"[historical_risk_patch] Test samples: {len(test_starts)}")

    pos_weight, positive_ratio = estimate_positive_weight(
        records,
        train_starts,
        args.seq_length,
        args.native_width,
        args.native_height,
        args.label_dilation_kernel,
        args.label_blur_radius,
        args.max_pos_weight,
    )
    print(f"[historical_risk_patch] Positive mass ratio (train): {positive_ratio:.8f}")
    print(f"[historical_risk_patch] Positive class weight: {pos_weight:.4f}")

    train_entries, train_patch_stats = build_patch_entries(
        records,
        train_starts,
        seq_length=args.seq_length,
        native_width=args.native_width,
        native_height=args.native_height,
        label_dilation_kernel=args.label_dilation_kernel,
        label_blur_radius=args.label_blur_radius,
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        ground_truth_threshold=args.ground_truth_threshold,
        # `positive_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
        positive_patches=args.train_positive_patches,
        # `negative_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
        negative_patches=args.train_negative_patches,
        seed=args.seed,
    )
    val_entries, val_patch_stats = build_patch_entries(
        records,
        val_starts,
        seq_length=args.seq_length,
        native_width=args.native_width,
        native_height=args.native_height,
        label_dilation_kernel=args.label_dilation_kernel,
        label_blur_radius=args.label_blur_radius,
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        ground_truth_threshold=args.ground_truth_threshold,
        # `positive_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
        positive_patches=args.eval_positive_patches,
        # `negative_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
        negative_patches=args.eval_negative_patches,
        seed=args.seed + 1,
    )
    test_entries, test_patch_stats = build_patch_entries(
        records,
        test_starts,
        seq_length=args.seq_length,
        native_width=args.native_width,
        native_height=args.native_height,
        label_dilation_kernel=args.label_dilation_kernel,
        label_blur_radius=args.label_blur_radius,
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        ground_truth_threshold=args.ground_truth_threshold,
        # `positive_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
        positive_patches=args.eval_positive_patches,
        # `negative_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
        negative_patches=args.eval_negative_patches,
        seed=args.seed + 2,
    )

    print(f"[historical_risk_patch] Train patch entries: {len(train_entries)}")
    print(f"[historical_risk_patch] Val patch entries: {len(val_entries)}")
    print(f"[historical_risk_patch] Test patch entries: {len(test_entries)}")

    train_sequence = NativePatchSequence(
        records,
        train_entries,
        seq_length=args.seq_length,
        native_width=args.native_width,
        native_height=args.native_height,
        # `patch_width` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_width=args.patch_width,
        # `patch_height` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_height=args.patch_height,
        input_dilation_kernel=args.input_dilation_kernel,
        label_dilation_kernel=args.label_dilation_kernel,
        label_blur_radius=args.label_blur_radius,
        feature_stack=args.feature_stack,
        context_kernel=args.context_kernel,
        batch_size=args.batch_size,
        shuffle=True,
        enable_augmentation=args.enable_augmentation,
        seed=args.seed,
    )
    val_sequence = NativePatchSequence(
        records,
        val_entries,
        seq_length=args.seq_length,
        native_width=args.native_width,
        native_height=args.native_height,
        # `patch_width` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_width=args.patch_width,
        # `patch_height` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_height=args.patch_height,
        input_dilation_kernel=args.input_dilation_kernel,
        label_dilation_kernel=args.label_dilation_kernel,
        label_blur_radius=args.label_blur_radius,
        feature_stack=args.feature_stack,
        context_kernel=args.context_kernel,
        batch_size=args.batch_size,
        shuffle=False,
        enable_augmentation=False,
        seed=args.seed + 10,
    )
    test_sequence = NativePatchSequence(
        records,
        test_entries,
        seq_length=args.seq_length,
        native_width=args.native_width,
        native_height=args.native_height,
        # `patch_width` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_width=args.patch_width,
        # `patch_height` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_height=args.patch_height,
        input_dilation_kernel=args.input_dilation_kernel,
        label_dilation_kernel=args.label_dilation_kernel,
        label_blur_radius=args.label_blur_radius,
        feature_stack=args.feature_stack,
        context_kernel=args.context_kernel,
        batch_size=args.batch_size,
        shuffle=False,
        enable_augmentation=False,
        seed=args.seed + 20,
    )

    y_val = collect_targets(val_sequence)
    y_test = collect_targets(test_sequence)
    model = build_model(
        seq_length=args.seq_length,
        # `patch_height` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_height=args.patch_height,
        # `patch_width` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_width=args.patch_width,
        channels=feature_stack_channels(args.feature_stack),
        # Bobot kelas positif agar loss lebih memperhatikan piksel hotspot yang langka.
        pos_weight=pos_weight,
        loss_strategy=args.loss_strategy,
        context_kernel=args.context_kernel,
    )

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        ValidationSweepCallback(
            val_sequence,
            y_val,
            args.ground_truth_threshold,
            args.threshold_sweep_step,
        ),
        ModelCheckpoint(str(output_model), monitor="val_model_score", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_model_score", patience=8, restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    # Training dijalankan pada sequence/patch yang sudah disiapkan dari citra hotspot historis.
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Model menghasilkan probability map risiko hotspot dari input sequence.
    val_predictions = model.predict(val_sequence, verbose=0)
    best_threshold, val_sweep, best_val_metrics = sweep_thresholds(
        y_val,
        val_predictions,
        args.ground_truth_threshold,
        args.threshold_sweep_step,
    )

    # Model menghasilkan probability map risiko hotspot dari input sequence.
    test_predictions = model.predict(test_sequence, verbose=0)
    # `default_test_metrics` adalah nilai bawaan yang dipilih sesuai lingkungan lokal atau Colab.
    default_test_metrics = compute_metrics(y_test, test_predictions, args.threshold, args.ground_truth_threshold)
    # `best_test_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    best_test_metrics = compute_metrics(y_test, test_predictions, best_threshold, args.ground_truth_threshold)
    # `default_test_buffered_metrics` adalah nilai bawaan yang dipilih sesuai lingkungan lokal atau Colab.
    default_test_buffered_metrics = compute_buffered_metrics(
        y_test,
        test_predictions,
        args.threshold,
        args.ground_truth_threshold,
        args.evaluation_buffer_radius,
    )
    # `best_test_buffered_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    best_test_buffered_metrics = compute_buffered_metrics(
        y_test,
        test_predictions,
        best_threshold,
        args.ground_truth_threshold,
        args.evaluation_buffer_radius,
    )
    # `best_val_buffered_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    best_val_buffered_metrics = compute_buffered_metrics(
        y_val,
        val_predictions,
        best_threshold,
        args.ground_truth_threshold,
        args.evaluation_buffer_radius,
    )

    report = {
        "mode": "historical_risk_patch_1528x773",
        "dataset_dir": str(dataset_dir),
        "image_extensions": list(image_extensions),
        "dataset_frame_count": len(records),
        "skipped_frame_count": len(skipped_records),
        "skipped_records_preview": skipped_records[:20],
        "date_start": records[0].date.strftime("%Y-%m-%d"),
        "date_end": records[-1].date.strftime("%Y-%m-%d"),
        "native_width": args.native_width,
        "native_height": args.native_height,
        "patch_width": args.patch_width,
        "patch_height": args.patch_height,
        "seq_length": args.seq_length,
        "channels": feature_stack_channels(args.feature_stack),
        "feature_stack": args.feature_stack,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "default_threshold": args.threshold,
        "ground_truth_threshold": args.ground_truth_threshold,
        "selected_threshold": best_threshold,
        "threshold_sweep_step": args.threshold_sweep_step,
        "legacy_dilation_kernel": args.dilation_kernel,
        "input_dilation_kernel": args.input_dilation_kernel,
        "label_dilation_kernel": args.label_dilation_kernel,
        "label_blur_radius": args.label_blur_radius,
        "evaluation_buffer_radius": args.evaluation_buffer_radius,
        "context_kernel": args.context_kernel,
        "loss_strategy": args.loss_strategy,
        "train_augmentation_enabled": args.enable_augmentation,
        "positive_ratio_train": positive_ratio,
        "positive_class_weight": pos_weight,
        "train_samples": len(train_starts),
        "val_samples": len(val_starts),
        "test_samples": len(test_starts),
        "train_patch_stats": train_patch_stats,
        "val_patch_stats": val_patch_stats,
        "test_patch_stats": test_patch_stats,
        "history": history.history,
        "validation_threshold_sweep": val_sweep,
        "validation_model_score": ValidationSweepCallback._compute_model_score(best_val_metrics),
        "validation_best_metrics": best_val_metrics,
        "validation_buffered_metrics_best_threshold": best_val_buffered_metrics,
        "test_metrics_default_threshold": default_test_metrics,
        "test_metrics_best_threshold": best_test_metrics,
        "test_buffered_metrics_default_threshold": default_test_buffered_metrics,
        "test_buffered_metrics_best_threshold": best_test_buffered_metrics,
        "test_metrics": best_test_metrics,
        "output_model": str(output_model),
    }
    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    print(json.dumps(report["validation_best_metrics"], indent=2))
    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    print(json.dumps(report["test_metrics_best_threshold"], indent=2))
    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    print(json.dumps(report["test_buffered_metrics_best_threshold"], indent=2))
    print(f"[historical_risk_patch] Threshold terbaik dari validation: {best_threshold}")
    print(f"[historical_risk_patch] Model terbaik tersimpan di: {output_model}")
    print(f"[historical_risk_patch] Report training tersimpan di: {output_report}")
    # Hasil ini dikembalikan sebagai output fungsi `run_training` untuk tahap berikutnya.
    return report


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> int:
    maybe_mount_drive()
    configure_tensorflow_runtime()
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parse_args()
    output_model, output_report = resolve_output_paths(
        output_model=Path(args.output_model).resolve(),
        output_report=Path(args.output_report).resolve(),
        overwrite=args.overwrite,
    )
    run_training(args, output_model, output_report)
    # Hasil ini dikembalikan sebagai output fungsi `main` untuk tahap berikutnya.
    return 0


if __name__ == "__main__" and not is_notebook_runtime():
    raise SystemExit(main())
