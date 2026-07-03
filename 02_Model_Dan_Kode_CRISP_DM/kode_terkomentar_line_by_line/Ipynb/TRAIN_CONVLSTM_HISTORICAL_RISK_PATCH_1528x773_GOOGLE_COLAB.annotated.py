# File anotasi dari `Ipynb/TRAIN_CONVLSTM_HISTORICAL_RISK_PATCH_1528x773_GOOGLE_COLAB.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Script training utama model historical_risk_patch berbasis dataset citra hotspot PNG.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Menjaga cara baca kode Python tetap cocok dengan versi yang digunakan.
from __future__ import annotations

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import argparse
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import json
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import math
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import random
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import re
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from dataclasses import dataclass
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from datetime import datetime, timedelta, timezone
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from functools import lru_cache
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from zoneinfo import ZoneInfo

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
RUNNING_IN_COLAB = False

# Mencoba menjalankan proses yang mungkin gagal.
try:
    # Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
    from google.colab import drive  # type: ignore

    # Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
    RUNNING_IN_COLAB = True
# Menangani kesalahan agar program tidak langsung berhenti.
except Exception:
    # Menyimpan nilai ke `drive` untuk dipakai pada langkah berikutnya.
    drive = None  # type: ignore

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import tensorflow as tf
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image, ImageFilter
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras import Sequential
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Dropout, Input


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# Membuat langkah kerja bernama `discover_project_root`.
def discover_project_root() -> Path:
    # Menyimpan nilai ke `base_candidates` untuk dipakai pada langkah berikutnya.
    base_candidates: list[Path] = []
    # Mengecek syarat sebelum melanjutkan proses.
    if "__file__" in globals():
        # Melanjutkan langkah kerja pada bagian kode ini.
        base_candidates.append(Path(__file__).resolve().parent)
    # Melanjutkan langkah kerja pada bagian kode ini.
    base_candidates.append(Path.cwd().resolve())

    # Mengulang proses untuk setiap data dalam daftar.
    for base in base_candidates:
        # Mengulang proses untuk setiap data dalam daftar.
        for candidate in [base, *base.parents]:
            # Mengecek syarat sebelum melanjutkan proses.
            if (candidate / "Ipynb").exists():
                # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
                return candidate
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return base_candidates[0]


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
PROJECT_ROOT = discover_project_root()
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
IPYNB_DIR = PROJECT_ROOT / "Ipynb" if (PROJECT_ROOT / "Ipynb").exists() else PROJECT_ROOT
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
LOCAL_DATASET_DIR = IPYNB_DIR / "Dataset History Fire Hotspot In Riau Province"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
LOCAL_PNG_DATASET_DIR = IPYNB_DIR / "Dataset History Fire Hotspot In Riau Province PNG"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
LOCAL_ARTIFACTS_DIR = IPYNB_DIR / "artifacts-native"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
COLAB_DATASET_DIR = Path(
    # Melanjutkan langkah kerja pada bagian kode ini.
    "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/data/dataset_riau/Dataset_History_Fire_Hotspot_In_Riau_Province"
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
COLAB_ALT_DATASET_DIR = Path(
    # Melanjutkan langkah kerja pada bagian kode ini.
    "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/data/Dataset_History_Fire_Hotspot_In_Riau_Province"
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
COLAB_OUTPUT_MODEL = Path(
    # Melanjutkan langkah kerja pada bagian kode ini.
    "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/artifacts-native/best_model_convlstm_historical_risk_patch_1528x773.keras"
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
COLAB_OUTPUT_REPORT = Path(
    # Melanjutkan langkah kerja pada bagian kode ini.
    "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/artifacts-native/convlstm_historical_risk_patch_1528x773_training_report.json"
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
LOCAL_OUTPUT_MODEL = LOCAL_ARTIFACTS_DIR / "best_model_convlstm_historical_risk_patch_1528x773.keras"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
LOCAL_OUTPUT_REPORT = LOCAL_ARTIFACTS_DIR / "convlstm_historical_risk_patch_1528x773_training_report.json"


# Membuat langkah kerja bernama `maybe_mount_drive`.
def maybe_mount_drive() -> None:
    # Mengecek syarat sebelum melanjutkan proses.
    if not RUNNING_IN_COLAB:
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return
    # Mencoba menjalankan proses yang mungkin gagal.
    try:
        # Melanjutkan langkah kerja pada bagian kode ini.
        drive.mount("/content/drive")  # type: ignore[attr-defined]
    # Menangani kesalahan agar program tidak langsung berhenti.
    except Exception as exc:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("Drive mount dilewati:", exc)


# Membuat langkah kerja bernama `configure_tensorflow_runtime`.
def configure_tensorflow_runtime() -> None:
    # Menyimpan nilai ke `gpus` untuk dipakai pada langkah berikutnya.
    gpus = tf.config.list_physical_devices("GPU")
    # Mengecek syarat sebelum melanjutkan proses.
    if gpus:
        # Mengulang proses untuk setiap data dalam daftar.
        for gpu in gpus:
            # Mencoba menjalankan proses yang mungkin gagal.
            try:
                # Melanjutkan langkah kerja pada bagian kode ini.
                tf.config.experimental.set_memory_growth(gpu, True)
            # Menangani kesalahan agar program tidak langsung berhenti.
            except Exception:
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                pass
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("GPU terdeteksi:", ", ".join(device.name for device in gpus))
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("GPU tidak terdeteksi. Training akan berjalan di CPU.")


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@dataclass(frozen=True)
# Membuat wadah bernama `DatasetRecord` untuk menyimpan data atau aturan kerja.
class DatasetRecord:
    # Menjelaskan data `path` yang disimpan atau dikirim pada bagian ini.
    path: Path
    # Menjelaskan data `date` yang disimpan atau dikirim pada bagian ini.
    date: datetime


# Membuat langkah kerja bernama `build_versioned_path`.
def build_versioned_path(path: Path, label: str) -> Path:
    # Mencoba menjalankan proses yang mungkin gagal.
    try:
        # Menyimpan nilai ke `wib_now` untuk dipakai pada langkah berikutnya.
        wib_now = datetime.now(ZoneInfo("Asia/Jakarta"))
    # Menangani kesalahan agar program tidak langsung berhenti.
    except Exception:
        # Menyimpan nilai ke `wib_now` untuk dipakai pada langkah berikutnya.
        wib_now = datetime.now(timezone(timedelta(hours=7)))
    # Menyimpan nilai ke `timestamp` untuk dipakai pada langkah berikutnya.
    timestamp = wib_now.strftime("%Y%m%d_%H%M%S_WIB")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return path.with_name(f"{path.stem}_{label}_{timestamp}{path.suffix}")


# Membuat langkah kerja bernama `resolve_output_paths`.
def resolve_output_paths(output_model: Path, output_report: Path, overwrite: bool) -> tuple[Path, Path]:
    # Mengecek syarat sebelum melanjutkan proses.
    if overwrite:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return output_model, output_report
    # Mengecek syarat sebelum melanjutkan proses.
    if not output_model.exists() and not output_report.exists():
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return output_model, output_report

    # Menyimpan nilai ke `versioned_model` untuk dipakai pada langkah berikutnya.
    versioned_model = build_versioned_path(output_model, "run")
    # Menyimpan nilai ke `versioned_report` untuk dipakai pada langkah berikutnya.
    versioned_report = build_versioned_path(output_report, "run")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("Artefak lama sudah ada. Menyimpan run baru tanpa menimpa file lama.")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Model baru: {versioned_model}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Report baru: {versioned_report}")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return versioned_model, versioned_report


# Membuat langkah kerja bernama `resolve_dataset_dir`.
def resolve_dataset_dir(dataset_dir_arg: str) -> Path:
    # Menyimpan nilai ke `direct_path` untuk dipakai pada langkah berikutnya.
    direct_path = Path(dataset_dir_arg).expanduser()
    # Menyimpan nilai ke `candidates` untuk dipakai pada langkah berikutnya.
    candidates = [direct_path]
    # Menyimpan nilai ke `variants` untuk dipakai pada langkah berikutnya.
    variants = [
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        LOCAL_DATASET_DIR,
        # Melanjutkan langkah kerja pada bagian kode ini.
        PROJECT_ROOT / "Dataset History Fire Hotspot In Riau Province",
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        COLAB_DATASET_DIR,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        COLAB_ALT_DATASET_DIR,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ]
    # Mengulang proses untuk setiap data dalam daftar.
    for candidate in variants:
        # Mengecek syarat sebelum melanjutkan proses.
        if candidate not in candidates:
            # Melanjutkan langkah kerja pada bagian kode ini.
            candidates.append(candidate)

    # Mengulang proses untuk setiap data dalam daftar.
    for candidate in candidates:
        # Mengecek syarat sebelum melanjutkan proses.
        if candidate.exists():
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return candidate.resolve()

    # Menyimpan nilai ke `searched` untuk dipakai pada langkah berikutnya.
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    # Menghentikan proses dan memberi pesan kesalahan yang jelas.
    raise FileNotFoundError(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Dataset tidak ditemukan. Path yang dicek:\n"
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"{searched}\n\n"
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Pastikan Google Drive sudah termount dan nama folder di Drive sesuai."
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )


# Membuat langkah kerja bernama `parse_args`.
def parse_args() -> argparse.Namespace:
    # Menyimpan nilai ke `default_dataset_dir` untuk dipakai pada langkah berikutnya.
    default_dataset_dir = COLAB_DATASET_DIR if RUNNING_IN_COLAB else LOCAL_DATASET_DIR
    # Menyimpan nilai ke `default_output_model` untuk dipakai pada langkah berikutnya.
    default_output_model = COLAB_OUTPUT_MODEL if RUNNING_IN_COLAB else LOCAL_OUTPUT_MODEL
    # Menyimpan nilai ke `default_output_report` untuk dipakai pada langkah berikutnya.
    default_output_report = COLAB_OUTPUT_REPORT if RUNNING_IN_COLAB else LOCAL_OUTPUT_REPORT
    # Menyimpan nilai ke `default_patch_size` untuk dipakai pada langkah berikutnya.
    default_patch_size = 192 if RUNNING_IN_COLAB else 160
    # Menyimpan nilai ke `default_batch_size` untuk dipakai pada langkah berikutnya.
    default_batch_size = 4 if RUNNING_IN_COLAB else 1
    # Menyimpan nilai ke `default_epochs` untuk dipakai pada langkah berikutnya.
    default_epochs = 30 if RUNNING_IN_COLAB else 20

    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description=(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Latih ConvLSTM native patch 1528x773 dengan target heatmap risiko "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "dari dataset hotspot JPG/JPEG/PNG."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--dataset-dir",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=str(default_dataset_dir),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--output-model",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=str(default_output_model),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--output-report",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=str(default_output_report),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--native-width", type=int, default=1528)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--native-height", type=int, default=773)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--patch-width", type=int, default=default_patch_size)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--patch-height", type=int, default=default_patch_size)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--seq-length", type=int, default=7)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--epochs", type=int, default=default_epochs)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--train-ratio", type=float, default=0.70)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--threshold", type=float, default=0.3)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--ground-truth-threshold", type=float, default=0.05)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--dilation-kernel", type=int, default=5)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--input-dilation-kernel",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=int,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=5,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Dilasi untuk mask input historis. Default 5 agar sama dengan baseline native patch.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--label-dilation-kernel",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=int,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=9,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Dilasi target risiko. Nilai lebih besar membentuk area risiko di sekitar hotspot.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--label-blur-radius",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=float,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=2.0,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Radius blur untuk membuat target density/heatmap lebih halus. Pakai 0 untuk menonaktifkan.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--evaluation-buffer-radius",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=int,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=5,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Radius buffer evaluasi berbasis toleransi spasial pada patch output.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--context-kernel", type=int, default=5)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--loss-strategy",
        # Menyimpan nilai ke `choices` untuk dipakai pada langkah berikutnya.
        choices=["wbce_dice", "wbce_dice_context"],
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="wbce_dice_context",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--threshold-sweep-step",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=float,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=0.05,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Langkah coarse threshold sweep validation.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--feature-stack", choices=["mask", "mask_context3"], default="mask")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--image-extensions",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=",".join(DEFAULT_IMAGE_EXTENSIONS),
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Daftar ekstensi gambar yang dibaca, pisahkan dengan koma. Contoh: .jpg,.jpeg,.png",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--train-positive-patches", type=int, default=3)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--train-negative-patches", type=int, default=1)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--eval-positive-patches", type=int, default=1)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--eval-negative-patches", type=int, default=1)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--enable-augmentation", action="store_true")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--seed", type=int, default=42)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--overwrite", action="store_true")
    # Menyimpan nilai ke `args, _` untuk dipakai pada langkah berikutnya.
    args, _ = parser.parse_known_args()
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return args


# Membuat langkah kerja bernama `is_notebook_runtime`.
def is_notebook_runtime() -> bool:
    # Mencoba menjalankan proses yang mungkin gagal.
    try:
        # Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
        from IPython import get_ipython  # type: ignore
    # Menangani kesalahan agar program tidak langsung berhenti.
    except Exception:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return False
    # Menyimpan nilai ke `shell` untuk dipakai pada langkah berikutnya.
    shell = get_ipython()
    # Mengecek syarat sebelum melanjutkan proses.
    if shell is None:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return False
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return shell.__class__.__name__ in {"ZMQInteractiveShell", "Shell"}


# Membuat langkah kerja bernama `feature_stack_channels`.
def feature_stack_channels(feature_stack: str) -> int:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return 3 if feature_stack == "mask_context3" else 1


# Membuat langkah kerja bernama `parse_image_extensions`.
def parse_image_extensions(value: str) -> tuple[str, ...]:
    # Menyimpan nilai ke `extensions` untuk dipakai pada langkah berikutnya.
    extensions: list[str] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for item in value.split(","):
        # Menyimpan nilai ke `ext` untuk dipakai pada langkah berikutnya.
        ext = item.strip().lower()
        # Mengecek syarat sebelum melanjutkan proses.
        if not ext:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Mengecek syarat sebelum melanjutkan proses.
        if not ext.startswith("."):
            # Menyimpan nilai ke `ext` untuk dipakai pada langkah berikutnya.
            ext = f".{ext}"
        # Mengecek syarat sebelum melanjutkan proses.
        if ext not in extensions:
            # Melanjutkan langkah kerja pada bagian kode ini.
            extensions.append(ext)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tuple(extensions) if extensions else DEFAULT_IMAGE_EXTENSIONS


# Membuat langkah kerja bernama `load_records`.
def load_records(dataset_dir: Path, image_extensions: tuple[str, ...]) -> list[DatasetRecord]:
    # Menyimpan nilai ke `records` untuk dipakai pada langkah berikutnya.
    records: list[DatasetRecord] = []
    # Menyimpan nilai ke `allowed_extensions` untuk dipakai pada langkah berikutnya.
    allowed_extensions = set(image_extensions)
    # Mengulang proses untuk setiap data dalam daftar.
    for path in sorted(item for item in dataset_dir.iterdir() if item.is_file()):
        # Mengecek syarat sebelum melanjutkan proses.
        if path.suffix.lower() not in allowed_extensions:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Menyimpan nilai ke `match` untuk dipakai pada langkah berikutnya.
        match = DATE_PATTERN.search(path.name)
        # Mengecek syarat sebelum melanjutkan proses.
        if not match:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Menyimpan nilai ke `records.append(DatasetRecord(path` untuk dipakai pada langkah berikutnya.
        records.append(DatasetRecord(path=path, date=datetime.strptime(match.group(1), "%Y-%m-%d")))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return sorted(records, key=lambda item: item.date)


# Membuat langkah kerja bernama `validate_records`.
def validate_records(records: list[DatasetRecord]) -> tuple[list[DatasetRecord], list[dict[str, str]]]:
    # Menyimpan nilai ke `valid_records` untuk dipakai pada langkah berikutnya.
    valid_records: list[DatasetRecord] = []
    # Menyimpan nilai ke `skipped_records` untuk dipakai pada langkah berikutnya.
    skipped_records: list[dict[str, str]] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for record in records:
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Mengecek syarat sebelum melanjutkan proses.
            if not record.path.exists():
                # Menghentikan proses dan memberi pesan kesalahan yang jelas.
                raise FileNotFoundError(f"File tidak ditemukan: {record.path}")
            # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
            with Image.open(record.path) as image:
                # Melanjutkan langkah kerja pada bagian kode ini.
                image.verify()
            # Melanjutkan langkah kerja pada bagian kode ini.
            valid_records.append(record)
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception as exc:
            # Melanjutkan langkah kerja pada bagian kode ini.
            skipped_records.append({"path": str(record.path), "reason": str(exc)})
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return valid_records, skipped_records


# Membuat langkah kerja bernama `_normalize_kernel`.
def _normalize_kernel(size: int) -> int:
    # Menyimpan nilai ke `size` untuk dipakai pada langkah berikutnya.
    size = max(1, int(size))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return size if size % 2 == 1 else size + 1

# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@lru_cache(maxsize=96)
# Membuat langkah kerja bernama `load_native_mask`.
def load_native_mask(
    # Menjelaskan data `path_str` yang disimpan atau dikirim pada bagian ini.
    path_str: str,
    # Menjelaskan data `native_width` yang disimpan atau dikirim pada bagian ini.
    native_width: int,
    # Menjelaskan data `native_height` yang disimpan atau dikirim pada bagian ini.
    native_height: int,
    # Menjelaskan data `dilation_kernel` yang disimpan atau dikirim pada bagian ini.
    dilation_kernel: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> np.ndarray:
    # Menyimpan nilai ke `dilation_kernel` untuk dipakai pada langkah berikutnya.
    dilation_kernel = _normalize_kernel(dilation_kernel)
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with Image.open(path_str) as image:
        # Mengubah citra ke warna RGB agar format gambar seragam.
        rgb = image.convert("RGB")
        # Mengubah warna RGB ke HSV agar warna merah hotspot lebih mudah dipisahkan.
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)

    # Mengambil komponen warna H (hue) untuk mengenali warna merah.
    h = hsv[:, :, 0]
    # Mengambil komponen S (saturation) untuk memastikan warna cukup kuat.
    s = hsv[:, :, 1]
    # Mengambil komponen V (brightness) untuk memastikan piksel cukup terang.
    v = hsv[:, :, 2]
    # Menandai piksel merah pada rentang warna merah bagian bawah.
    red_low = (h <= 14) & (s >= 70) & (v >= 50)
    # Menandai piksel merah pada rentang warna merah bagian atas.
    red_high = (h >= 242) & (s >= 70) & (v >= 50)
    # Menggabungkan hasil deteksi merah menjadi mask hotspot.
    mask = ((red_low | red_high).astype(np.uint8)) * 255

    # Mengubah mask angka menjadi gambar hitam-putih.
    mask_image = Image.fromarray(mask)
    # Mengecek syarat sebelum melanjutkan proses.
    if dilation_kernel > 1:
        # Memperbesar titik hotspot kecil agar tidak mudah hilang.
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=dilation_kernel))
    # Mengecek syarat sebelum melanjutkan proses.
    if mask_image.size != (native_width, native_height):
        # Menyimpan nilai ke `mask_image` untuk dipakai pada langkah berikutnya.
        mask_image = mask_image.resize((native_width, native_height), Image.BILINEAR)

    # Menyimpan nilai ke `density` untuk dipakai pada langkah berikutnya.
    density = np.asarray(mask_image, dtype=np.float32) / 255.0
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.clip(density, 0.0, 1.0)


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@lru_cache(maxsize=96)
# Membuat langkah kerja bernama `load_native_risk_map`.
def load_native_risk_map(
    # Menjelaskan data `path_str` yang disimpan atau dikirim pada bagian ini.
    path_str: str,
    # Menjelaskan data `native_width` yang disimpan atau dikirim pada bagian ini.
    native_width: int,
    # Menjelaskan data `native_height` yang disimpan atau dikirim pada bagian ini.
    native_height: int,
    # Menjelaskan data `label_dilation_kernel` yang disimpan atau dikirim pada bagian ini.
    label_dilation_kernel: int,
    # Menjelaskan data `label_blur_radius` yang disimpan atau dikirim pada bagian ini.
    label_blur_radius: float,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> np.ndarray:
    # Catatan pembuka/penutup yang menjelaskan isi bagian kode.
    """Build a softer target map so hotspot labels represent nearby risk areas."""
    # Menyimpan nilai ke `label_dilation_kernel` untuk dipakai pada langkah berikutnya.
    label_dilation_kernel = _normalize_kernel(label_dilation_kernel)
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with Image.open(path_str) as image:
        # Mengubah citra ke warna RGB agar format gambar seragam.
        rgb = image.convert("RGB")
        # Mengubah warna RGB ke HSV agar warna merah hotspot lebih mudah dipisahkan.
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)

    # Mengambil komponen warna H (hue) untuk mengenali warna merah.
    h = hsv[:, :, 0]
    # Mengambil komponen S (saturation) untuk memastikan warna cukup kuat.
    s = hsv[:, :, 1]
    # Mengambil komponen V (brightness) untuk memastikan piksel cukup terang.
    v = hsv[:, :, 2]
    # Menandai piksel merah pada rentang warna merah bagian bawah.
    red_low = (h <= 14) & (s >= 70) & (v >= 50)
    # Menandai piksel merah pada rentang warna merah bagian atas.
    red_high = (h >= 242) & (s >= 70) & (v >= 50)
    # Menggabungkan hasil deteksi merah menjadi mask hotspot.
    mask = ((red_low | red_high).astype(np.uint8)) * 255

    # Menyimpan nilai ke `risk_image` untuk dipakai pada langkah berikutnya.
    risk_image = Image.fromarray(mask)
    # Mengecek syarat sebelum melanjutkan proses.
    if label_dilation_kernel > 1:
        # Menyimpan nilai ke `risk_image` untuk dipakai pada langkah berikutnya.
        risk_image = risk_image.filter(ImageFilter.MaxFilter(size=label_dilation_kernel))
    # Mengecek syarat sebelum melanjutkan proses.
    if label_blur_radius > 0:
        # Menyimpan nilai ke `risk_image` untuk dipakai pada langkah berikutnya.
        risk_image = risk_image.filter(ImageFilter.GaussianBlur(radius=float(label_blur_radius)))
    # Mengecek syarat sebelum melanjutkan proses.
    if risk_image.size != (native_width, native_height):
        # Menyimpan nilai ke `risk_image` untuk dipakai pada langkah berikutnya.
        risk_image = risk_image.resize((native_width, native_height), Image.BILINEAR)

    # Menyimpan nilai ke `density` untuk dipakai pada langkah berikutnya.
    density = np.asarray(risk_image, dtype=np.float32) / 255.0
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.clip(density, 0.0, 1.0)


# Membuat langkah kerja bernama `build_feature_map`.
def build_feature_map(mask: np.ndarray, feature_stack: str, context_kernel: int) -> np.ndarray:
    # Mengecek syarat sebelum melanjutkan proses.
    if feature_stack == "mask_context3":
        # Menyimpan nilai ke `context_kernel` untuk dipakai pada langkah berikutnya.
        context_kernel = _normalize_kernel(context_kernel)
        # Menyimpan nilai ke `context_image` untuk dipakai pada langkah berikutnya.
        context_image = Image.fromarray(np.uint8(mask * 255.0))
        # Mengecek syarat sebelum melanjutkan proses.
        if context_kernel > 1:
            # Menyimpan nilai ke `context_image` untuk dipakai pada langkah berikutnya.
            context_image = context_image.filter(ImageFilter.BoxBlur(radius=max(1, context_kernel // 2)))
        # Menyimpan nilai ke `context_map` untuk dipakai pada langkah berikutnya.
        context_map = np.asarray(context_image, dtype=np.float32) / 255.0
        # Menyimpan nilai ke `hotspot_core` untuk dipakai pada langkah berikutnya.
        hotspot_core = (mask >= 0.25).astype(np.float32)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.stack([mask, np.clip(context_map, 0.0, 1.0), hotspot_core], axis=-1).astype(np.float32)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.expand_dims(mask.astype(np.float32), axis=-1)


# Membuat langkah kerja bernama `build_sample_starts`.
def build_sample_starts(record_count: int, seq_length: int) -> list[int]:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return list(range(record_count - seq_length))


# Membuat langkah kerja bernama `split_sample_starts`.
def split_sample_starts(sample_starts: list[int], train_ratio: float, val_ratio: float) -> tuple[list[int], list[int], list[int]]:
    # Menyimpan nilai ke `sample_count` untuk dipakai pada langkah berikutnya.
    sample_count = len(sample_starts)
    # Menyimpan nilai ke `train_end` untuk dipakai pada langkah berikutnya.
    train_end = max(1, int(sample_count * train_ratio))
    # Menyimpan nilai ke `val_end` untuk dipakai pada langkah berikutnya.
    val_end = max(train_end + 1, int(sample_count * (train_ratio + val_ratio)))
    # Menyimpan nilai ke `val_end` untuk dipakai pada langkah berikutnya.
    val_end = min(val_end, sample_count - 1)
    # Menyimpan nilai ke `train` untuk dipakai pada langkah berikutnya.
    train = sample_starts[:train_end]
    # Menyimpan nilai ke `val` untuk dipakai pada langkah berikutnya.
    val = sample_starts[train_end:val_end]
    # Menyimpan nilai ke `test` untuk dipakai pada langkah berikutnya.
    test = sample_starts[val_end:]
    # Mengecek syarat sebelum melanjutkan proses.
    if not val or not test:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Split train/val/test gagal. Periksa rasio dataset.")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return train, val, test


# Membuat langkah kerja bernama `sample_patch_centers`.
def sample_patch_centers(
    # Menjelaskan data `target_mask` yang disimpan atau dikirim pada bagian ini.
    target_mask: np.ndarray,
    # Menjelaskan data `positive_patch_count` yang disimpan atau dikirim pada bagian ini.
    positive_patch_count: int,
    # Menjelaskan data `negative_patch_count` yang disimpan atau dikirim pada bagian ini.
    negative_patch_count: int,
    # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
    ground_truth_threshold: float,
    # Menjelaskan data `rng` yang disimpan atau dikirim pada bagian ini.
    rng: np.random.Generator,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> list[tuple[int, int]]:
    # Menyimpan nilai ke `binary` untuk dipakai pada langkah berikutnya.
    binary = target_mask >= ground_truth_threshold
    # Menyimpan nilai ke `positive_coords` untuk dipakai pada langkah berikutnya.
    positive_coords = np.argwhere(binary)
    # Menyimpan nilai ke `centers` untuk dipakai pada langkah berikutnya.
    centers: list[tuple[int, int]] = []

    # Mengecek syarat sebelum melanjutkan proses.
    if len(positive_coords) > 0 and positive_patch_count > 0:
        # Menyimpan nilai ke `replace` untuk dipakai pada langkah berikutnya.
        replace = len(positive_coords) < positive_patch_count
        # Menyimpan nilai ke `indices` untuk dipakai pada langkah berikutnya.
        indices = rng.choice(len(positive_coords), size=positive_patch_count, replace=replace)
        # Mengulang proses untuk setiap data dalam daftar.
        for idx in np.atleast_1d(indices):
            # Menyimpan nilai ke `cy, cx` untuk dipakai pada langkah berikutnya.
            cy, cx = positive_coords[int(idx)]
            # Melanjutkan langkah kerja pada bagian kode ini.
            centers.append((int(cy), int(cx)))

    # Menyimpan nilai ke `neg_needed` untuk dipakai pada langkah berikutnya.
    neg_needed = negative_patch_count
    # Menyimpan nilai ke `attempts` untuk dipakai pada langkah berikutnya.
    attempts = max(neg_needed * 40, 40)
    # Mengulang proses selama syaratnya masih terpenuhi.
    while neg_needed > 0 and attempts > 0:
        # Menyimpan nilai ke `cy` untuk dipakai pada langkah berikutnya.
        cy = int(rng.integers(0, target_mask.shape[0]))
        # Menyimpan nilai ke `cx` untuk dipakai pada langkah berikutnya.
        cx = int(rng.integers(0, target_mask.shape[1]))
        # Mengecek syarat sebelum melanjutkan proses.
        if not binary[cy, cx]:
            # Melanjutkan langkah kerja pada bagian kode ini.
            centers.append((cy, cx))
            # Menyimpan nilai ke `neg_needed -` untuk dipakai pada langkah berikutnya.
            neg_needed -= 1
        # Menyimpan nilai ke `attempts -` untuk dipakai pada langkah berikutnya.
        attempts -= 1

    # Mengecek syarat sebelum melanjutkan proses.
    if not centers:
        # Melanjutkan langkah kerja pada bagian kode ini.
        centers.append((target_mask.shape[0] // 2, target_mask.shape[1] // 2))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return centers


# Membuat langkah kerja bernama `extract_patch`.
def extract_patch(array: np.ndarray, cy: int, cx: int, patch_height: int, patch_width: int) -> np.ndarray:
    # Mengecek syarat sebelum melanjutkan proses.
    if array.ndim == 2:
        # Menyimpan nilai ke `array` untuk dipakai pada langkah berikutnya.
        array = array[..., None]

    # Menyimpan nilai ke `height, width` untuk dipakai pada langkah berikutnya.
    height, width = array.shape[:2]
    # Menyimpan nilai ke `half_h` untuk dipakai pada langkah berikutnya.
    half_h = patch_height // 2
    # Menyimpan nilai ke `half_w` untuk dipakai pada langkah berikutnya.
    half_w = patch_width // 2
    # Menyimpan nilai ke `top` untuk dipakai pada langkah berikutnya.
    top = cy - half_h
    # Menyimpan nilai ke `left` untuk dipakai pada langkah berikutnya.
    left = cx - half_w
    # Menyimpan nilai ke `bottom` untuk dipakai pada langkah berikutnya.
    bottom = top + patch_height
    # Menyimpan nilai ke `right` untuk dipakai pada langkah berikutnya.
    right = left + patch_width

    # Menyimpan nilai ke `pad_top` untuk dipakai pada langkah berikutnya.
    pad_top = max(0, -top)
    # Menyimpan nilai ke `pad_left` untuk dipakai pada langkah berikutnya.
    pad_left = max(0, -left)
    # Menyimpan nilai ke `pad_bottom` untuk dipakai pada langkah berikutnya.
    pad_bottom = max(0, bottom - height)
    # Menyimpan nilai ke `pad_right` untuk dipakai pada langkah berikutnya.
    pad_right = max(0, right - width)

    # Mengecek syarat sebelum melanjutkan proses.
    if pad_top or pad_left or pad_bottom or pad_right:
        # Menyimpan nilai ke `array` untuk dipakai pada langkah berikutnya.
        array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
        # Menyimpan nilai ke `top +` untuk dipakai pada langkah berikutnya.
        top += pad_top
        # Menyimpan nilai ke `left +` untuk dipakai pada langkah berikutnya.
        left += pad_left
        # Menyimpan nilai ke `bottom` untuk dipakai pada langkah berikutnya.
        bottom = top + patch_height
        # Menyimpan nilai ke `right` untuk dipakai pada langkah berikutnya.
        right = left + patch_width

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return array[top:bottom, left:right, :].astype(np.float32)


# Membuat langkah kerja bernama `estimate_positive_weight`.
def estimate_positive_weight(
    # Menjelaskan data `records` yang disimpan atau dikirim pada bagian ini.
    records: list[DatasetRecord],
    # Menjelaskan data `sample_starts` yang disimpan atau dikirim pada bagian ini.
    sample_starts: list[int],
    # Menjelaskan data `seq_length` yang disimpan atau dikirim pada bagian ini.
    seq_length: int,
    # Menjelaskan data `native_width` yang disimpan atau dikirim pada bagian ini.
    native_width: int,
    # Menjelaskan data `native_height` yang disimpan atau dikirim pada bagian ini.
    native_height: int,
    # Menjelaskan data `label_dilation_kernel` yang disimpan atau dikirim pada bagian ini.
    label_dilation_kernel: int,
    # Menjelaskan data `label_blur_radius` yang disimpan atau dikirim pada bagian ini.
    label_blur_radius: float,
    # Menjelaskan data `max_pos_weight` yang disimpan atau dikirim pada bagian ini.
    max_pos_weight: float,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tuple[float, float]:
    # Menyimpan nilai ke `positive_sum` untuk dipakai pada langkah berikutnya.
    positive_sum = 0.0
    # Menyimpan nilai ke `pixel_count` untuk dipakai pada langkah berikutnya.
    pixel_count = 0
    # Mengulang proses untuk setiap data dalam daftar.
    for start in sample_starts:
        # Menggabungkan hasil deteksi merah menjadi mask hotspot.
        mask = load_native_risk_map(
            # Melanjutkan langkah kerja pada bagian kode ini.
            str(records[start + seq_length].path),
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            native_width,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            native_height,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            label_dilation_kernel,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            label_blur_radius,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `positive_sum +` untuk dipakai pada langkah berikutnya.
        positive_sum += float(mask.sum())
        # Menyimpan nilai ke `pixel_count +` untuk dipakai pada langkah berikutnya.
        pixel_count += mask.size
    # Menyimpan nilai ke `positive_ratio` untuk dipakai pada langkah berikutnya.
    positive_ratio = max(positive_sum / max(pixel_count, 1), 1e-6)
    # Menyimpan nilai ke `pos_weight` untuk dipakai pada langkah berikutnya.
    pos_weight = min((1.0 - positive_ratio) / positive_ratio, max_pos_weight)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return float(max(pos_weight, 1.0)), float(positive_ratio)


# Membuat langkah kerja bernama `_compute_weighted_bce`.
def _compute_weighted_bce(y_true: tf.Tensor, y_pred: tf.Tensor, pos_weight_tensor: tf.Tensor) -> tf.Tensor:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tf.reduce_mean(
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        -(
            # Melanjutkan langkah kerja pada bagian kode ini.
            (pos_weight_tensor * y_true * tf.math.log(y_pred))
            # Melanjutkan langkah kerja pada bagian kode ini.
            + ((1.0 - y_true) * tf.math.log(1.0 - y_pred))
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )


# Membuat langkah kerja bernama `_compute_dice_loss`.
def _compute_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # Menyimpan nilai ke `intersection` untuk dipakai pada langkah berikutnya.
    intersection = tf.reduce_sum(y_true * y_pred)
    # Menyimpan nilai ke `denominator` untuk dipakai pada langkah berikutnya.
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))


# Membuat langkah kerja bernama `_compute_context_bce`.
def _compute_context_bce(y_true: tf.Tensor, y_pred: tf.Tensor, context_kernel: int) -> tf.Tensor:
    # Menyimpan nilai ke `pooled_context` untuk dipakai pada langkah berikutnya.
    pooled_context = tf.nn.max_pool2d(y_true, ksize=context_kernel, strides=1, padding="SAME")
    # Menyimpan nilai ke `context_weight` untuk dipakai pada langkah berikutnya.
    context_weight = 1.0 + (2.0 * pooled_context)
    # Menyimpan nilai ke `context_bce_map` untuk dipakai pada langkah berikutnya.
    context_bce_map = -(
        # Melanjutkan langkah kerja pada bagian kode ini.
        (y_true * tf.math.log(y_pred))
        # Melanjutkan langkah kerja pada bagian kode ini.
        + ((1.0 - y_true) * tf.math.log(1.0 - y_pred))
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tf.reduce_mean(context_weight * context_bce_map)


# Membuat langkah kerja bernama `make_weighted_bce_dice_loss`.
def make_weighted_bce_dice_loss(pos_weight: float):
    # Menyimpan nilai ke `pos_weight_tensor` untuk dipakai pada langkah berikutnya.
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    # Menyimpan nilai ke `epsilon` untuk dipakai pada langkah berikutnya.
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Membuat langkah kerja bernama `loss`.
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Menyimpan nilai ke `y_true` untuk dipakai pada langkah berikutnya.
        y_true = tf.cast(y_true, tf.float32)
        # Menyimpan nilai ke `y_pred` untuk dipakai pada langkah berikutnya.
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        # Menyimpan nilai ke `weighted_bce` untuk dipakai pada langkah berikutnya.
        weighted_bce = _compute_weighted_bce(y_true, y_pred, pos_weight_tensor)
        # Menyimpan nilai ke `dice_loss` untuk dipakai pada langkah berikutnya.
        dice_loss = _compute_dice_loss(y_true, y_pred)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return (0.7 * weighted_bce) + (0.3 * dice_loss)

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return loss


# Membuat langkah kerja bernama `make_weighted_bce_dice_context_loss`.
def make_weighted_bce_dice_context_loss(pos_weight: float, context_kernel: int):
    # Menyimpan nilai ke `pos_weight_tensor` untuk dipakai pada langkah berikutnya.
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    # Menyimpan nilai ke `context_kernel` untuk dipakai pada langkah berikutnya.
    context_kernel = _normalize_kernel(context_kernel)
    # Menyimpan nilai ke `epsilon` untuk dipakai pada langkah berikutnya.
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Membuat langkah kerja bernama `loss`.
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Menyimpan nilai ke `y_true` untuk dipakai pada langkah berikutnya.
        y_true = tf.cast(y_true, tf.float32)
        # Menyimpan nilai ke `y_pred` untuk dipakai pada langkah berikutnya.
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        # Menyimpan nilai ke `weighted_bce` untuk dipakai pada langkah berikutnya.
        weighted_bce = _compute_weighted_bce(y_true, y_pred, pos_weight_tensor)
        # Menyimpan nilai ke `dice_loss` untuk dipakai pada langkah berikutnya.
        dice_loss = _compute_dice_loss(y_true, y_pred)
        # Menyimpan nilai ke `context_bce` untuk dipakai pada langkah berikutnya.
        context_bce = _compute_context_bce(y_true, y_pred, context_kernel)
        # Menyimpan nilai ke `base_loss` untuk dipakai pada langkah berikutnya.
        base_loss = (0.7 * weighted_bce) + (0.3 * dice_loss)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return (0.75 * base_loss) + (0.25 * context_bce)

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return loss


# Membuat langkah kerja bernama `build_loss`.
def build_loss(
    # Menjelaskan data `loss_strategy` yang disimpan atau dikirim pada bagian ini.
    loss_strategy: str,
    # Menjelaskan data `pos_weight` yang disimpan atau dikirim pada bagian ini.
    pos_weight: float,
    # Menjelaskan data `context_kernel` yang disimpan atau dikirim pada bagian ini.
    context_kernel: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
):
    # Mengecek syarat sebelum melanjutkan proses.
    if loss_strategy == "wbce_dice_context":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return make_weighted_bce_dice_context_loss(pos_weight, context_kernel)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return make_weighted_bce_dice_loss(pos_weight)


# Membuat langkah kerja bernama `build_model`.
def build_model(
    # Menjelaskan data `seq_length` yang disimpan atau dikirim pada bagian ini.
    seq_length: int,
    # Menjelaskan data `patch_height` yang disimpan atau dikirim pada bagian ini.
    patch_height: int,
    # Menjelaskan data `patch_width` yang disimpan atau dikirim pada bagian ini.
    patch_width: int,
    # Menjelaskan data `channels` yang disimpan atau dikirim pada bagian ini.
    channels: int,
    # Menjelaskan data `pos_weight` yang disimpan atau dikirim pada bagian ini.
    pos_weight: float,
    # Menjelaskan data `loss_strategy` yang disimpan atau dikirim pada bagian ini.
    loss_strategy: str,
    # Menjelaskan data `context_kernel` yang disimpan atau dikirim pada bagian ini.
    context_kernel: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tf.keras.Model:
    # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
    model = Sequential(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            # Menyimpan nilai ke `Input(shape` untuk dipakai pada langkah berikutnya.
            Input(shape=(seq_length, patch_height, patch_width, channels)),
            # Menyimpan nilai ke `ConvLSTM2D(filters` untuk dipakai pada langkah berikutnya.
            ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True),
            # Melanjutkan langkah kerja pada bagian kode ini.
            BatchNormalization(),
            # Melanjutkan langkah kerja pada bagian kode ini.
            Dropout(0.2),
            # Menyimpan nilai ke `ConvLSTM2D(filters` untuk dipakai pada langkah berikutnya.
            ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=False),
            # Melanjutkan langkah kerja pada bagian kode ini.
            BatchNormalization(),
            # Melanjutkan langkah kerja pada bagian kode ini.
            Dropout(0.2),
            # Menyimpan nilai ke `Conv2D(filters` untuk dipakai pada langkah berikutnya.
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
            # Menyimpan nilai ke `Conv2D(filters` untuk dipakai pada langkah berikutnya.
            Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same"),
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    model.compile(
        # Menyimpan nilai ke `optimizer` untuk dipakai pada langkah berikutnya.
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # Menyimpan nilai ke `loss` untuk dipakai pada langkah berikutnya.
        loss=build_loss(loss_strategy, pos_weight, context_kernel),
        # Menyimpan nilai ke `jit_compile` untuk dipakai pada langkah berikutnya.
        jit_compile=False,
        # Menyimpan nilai ke `metrics` untuk dipakai pada langkah berikutnya.
        metrics=[
            # Menyimpan nilai ke `tf.keras.metrics.BinaryAccuracy(name` untuk dipakai pada langkah berikutnya.
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            # Menyimpan nilai ke `tf.keras.metrics.Precision(name` untuk dipakai pada langkah berikutnya.
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            # Menyimpan nilai ke `tf.keras.metrics.Recall(name` untuk dipakai pada langkah berikutnya.
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
            # Menyimpan nilai ke `tf.keras.metrics.AUC(name` untuk dipakai pada langkah berikutnya.
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ],
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return model


# Membuat wadah bernama `NativePatchSequence` untuk menyimpan data atau aturan kerja.
class NativePatchSequence(tf.keras.utils.Sequence):
    # Membuat langkah kerja bernama `__init__`.
    def __init__(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        self,
        # Menjelaskan data `records` yang disimpan atau dikirim pada bagian ini.
        records: list[DatasetRecord],
        # Menjelaskan data `entries` yang disimpan atau dikirim pada bagian ini.
        entries: list[tuple[int, int, int]],
        # Melanjutkan langkah kerja pada bagian kode ini.
        *,
        # Menjelaskan data `seq_length` yang disimpan atau dikirim pada bagian ini.
        seq_length: int,
        # Menjelaskan data `native_width` yang disimpan atau dikirim pada bagian ini.
        native_width: int,
        # Menjelaskan data `native_height` yang disimpan atau dikirim pada bagian ini.
        native_height: int,
        # Menjelaskan data `patch_width` yang disimpan atau dikirim pada bagian ini.
        patch_width: int,
        # Menjelaskan data `patch_height` yang disimpan atau dikirim pada bagian ini.
        patch_height: int,
        # Menjelaskan data `input_dilation_kernel` yang disimpan atau dikirim pada bagian ini.
        input_dilation_kernel: int,
        # Menjelaskan data `label_dilation_kernel` yang disimpan atau dikirim pada bagian ini.
        label_dilation_kernel: int,
        # Menjelaskan data `label_blur_radius` yang disimpan atau dikirim pada bagian ini.
        label_blur_radius: float,
        # Menjelaskan data `feature_stack` yang disimpan atau dikirim pada bagian ini.
        feature_stack: str,
        # Menjelaskan data `context_kernel` yang disimpan atau dikirim pada bagian ini.
        context_kernel: int,
        # Menjelaskan data `batch_size` yang disimpan atau dikirim pada bagian ini.
        batch_size: int,
        # Menjelaskan data `shuffle` yang disimpan atau dikirim pada bagian ini.
        shuffle: bool,
        # Menjelaskan data `enable_augmentation` yang disimpan atau dikirim pada bagian ini.
        enable_augmentation: bool,
        # Menjelaskan data `seed` yang disimpan atau dikirim pada bagian ini.
        seed: int,
        # Melanjutkan langkah kerja pada bagian kode ini.
        **kwargs,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ) -> None:
        # Melanjutkan langkah kerja pada bagian kode ini.
        super().__init__(**kwargs)
        # Menyimpan nilai ke `self.records` untuk dipakai pada langkah berikutnya.
        self.records = records
        # Menyimpan nilai ke `self.entries` untuk dipakai pada langkah berikutnya.
        self.entries = list(entries)
        # Menyimpan nilai ke `self.seq_length` untuk dipakai pada langkah berikutnya.
        self.seq_length = seq_length
        # Menyimpan nilai ke `self.native_width` untuk dipakai pada langkah berikutnya.
        self.native_width = native_width
        # Menyimpan nilai ke `self.native_height` untuk dipakai pada langkah berikutnya.
        self.native_height = native_height
        # Menyimpan nilai ke `self.patch_width` untuk dipakai pada langkah berikutnya.
        self.patch_width = patch_width
        # Menyimpan nilai ke `self.patch_height` untuk dipakai pada langkah berikutnya.
        self.patch_height = patch_height
        # Menyimpan nilai ke `self.input_dilation_kernel` untuk dipakai pada langkah berikutnya.
        self.input_dilation_kernel = input_dilation_kernel
        # Menyimpan nilai ke `self.label_dilation_kernel` untuk dipakai pada langkah berikutnya.
        self.label_dilation_kernel = label_dilation_kernel
        # Menyimpan nilai ke `self.label_blur_radius` untuk dipakai pada langkah berikutnya.
        self.label_blur_radius = label_blur_radius
        # Menyimpan nilai ke `self.feature_stack` untuk dipakai pada langkah berikutnya.
        self.feature_stack = feature_stack
        # Menyimpan nilai ke `self.context_kernel` untuk dipakai pada langkah berikutnya.
        self.context_kernel = context_kernel
        # Menyimpan nilai ke `self.batch_size` untuk dipakai pada langkah berikutnya.
        self.batch_size = batch_size
        # Menyimpan nilai ke `self.shuffle` untuk dipakai pada langkah berikutnya.
        self.shuffle = shuffle
        # Menyimpan nilai ke `self.enable_augmentation` untuk dipakai pada langkah berikutnya.
        self.enable_augmentation = enable_augmentation
        # Menyimpan nilai ke `self.channels` untuk dipakai pada langkah berikutnya.
        self.channels = feature_stack_channels(feature_stack)
        # Menyimpan nilai ke `self._rng` untuk dipakai pada langkah berikutnya.
        self._rng = random.Random(seed)
        # Melanjutkan langkah kerja pada bagian kode ini.
        self.on_epoch_end()

    # Membuat langkah kerja bernama `__len__`.
    def __len__(self) -> int:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return math.ceil(len(self.entries) / self.batch_size)

    # Membuat langkah kerja bernama `__getitem__`.
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        # Menyimpan nilai ke `batch_entries` untuk dipakai pada langkah berikutnya.
        batch_entries = self.entries[index * self.batch_size : (index + 1) * self.batch_size]
        # Menyimpan nilai ke `x_batch` untuk dipakai pada langkah berikutnya.
        x_batch = np.zeros(
            # Melanjutkan langkah kerja pada bagian kode ini.
            (len(batch_entries), self.seq_length, self.patch_height, self.patch_width, self.channels),
            # Menyimpan nilai ke `dtype` untuk dipakai pada langkah berikutnya.
            dtype=np.float32,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `y_batch` untuk dipakai pada langkah berikutnya.
        y_batch = np.zeros((len(batch_entries), self.patch_height, self.patch_width, 1), dtype=np.float32)
        # Menyimpan nilai ke `sample_cache` untuk dipakai pada langkah berikutnya.
        sample_cache: dict[int, tuple[list[np.ndarray], np.ndarray]] = {}

        # Mengulang proses untuk setiap data dalam daftar.
        for item_index, (start, cy, cx) in enumerate(batch_entries):
            # Mengecek syarat sebelum melanjutkan proses.
            if start not in sample_cache:
                # Menyimpan nilai ke `sequence_features` untuk dipakai pada langkah berikutnya.
                sequence_features: list[np.ndarray] = []
                # Mengulang proses untuk setiap data dalam daftar.
                for offset in range(self.seq_length):
                    # Menggabungkan hasil deteksi merah menjadi mask hotspot.
                    mask = load_native_mask(
                        # Melanjutkan langkah kerja pada bagian kode ini.
                        str(self.records[start + offset].path),
                        # Melanjutkan langkah kerja pada bagian kode ini.
                        self.native_width,
                        # Melanjutkan langkah kerja pada bagian kode ini.
                        self.native_height,
                        # Melanjutkan langkah kerja pada bagian kode ini.
                        self.input_dilation_kernel,
                    # Menutup susunan data atau perintah yang dimulai sebelumnya.
                    )
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    sequence_features.append(build_feature_map(mask, self.feature_stack, self.context_kernel))

                # Menyimpan nilai ke `target_mask` untuk dipakai pada langkah berikutnya.
                target_mask = load_native_risk_map(
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    str(self.records[start + self.seq_length].path),
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    self.native_width,
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    self.native_height,
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    self.label_dilation_kernel,
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    self.label_blur_radius,
                # Menutup susunan data atau perintah yang dimulai sebelumnya.
                )
                # Menyimpan nilai ke `sample_cache[start]` untuk dipakai pada langkah berikutnya.
                sample_cache[start] = (sequence_features, target_mask[..., None].astype(np.float32))

            # Menyimpan nilai ke `sequence_features, target_mask` untuk dipakai pada langkah berikutnya.
            sequence_features, target_mask = sample_cache[start]
            # Menyimpan nilai ke `x_patch` untuk dipakai pada langkah berikutnya.
            x_patch = np.stack(
                # Melanjutkan langkah kerja pada bagian kode ini.
                [extract_patch(feature, cy, cx, self.patch_height, self.patch_width) for feature in sequence_features],
                # Menyimpan nilai ke `axis` untuk dipakai pada langkah berikutnya.
                axis=0,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
            # Menyimpan nilai ke `y_patch` untuk dipakai pada langkah berikutnya.
            y_patch = extract_patch(target_mask, cy, cx, self.patch_height, self.patch_width)

            # Mengecek syarat sebelum melanjutkan proses.
            if self.enable_augmentation:
                # Mengecek syarat sebelum melanjutkan proses.
                if self._rng.random() < 0.5:
                    # Menyimpan nilai ke `x_patch` untuk dipakai pada langkah berikutnya.
                    x_patch = np.flip(x_patch, axis=2)
                    # Menyimpan nilai ke `y_patch` untuk dipakai pada langkah berikutnya.
                    y_patch = np.flip(y_patch, axis=1)
                # Mengecek syarat sebelum melanjutkan proses.
                if self._rng.random() < 0.5:
                    # Menyimpan nilai ke `x_patch` untuk dipakai pada langkah berikutnya.
                    x_patch = np.flip(x_patch, axis=1)
                    # Menyimpan nilai ke `y_patch` untuk dipakai pada langkah berikutnya.
                    y_patch = np.flip(y_patch, axis=0)

            # Menyimpan nilai ke `x_batch[item_index]` untuk dipakai pada langkah berikutnya.
            x_batch[item_index] = x_patch
            # Menyimpan nilai ke `y_batch[item_index]` untuk dipakai pada langkah berikutnya.
            y_batch[item_index] = y_patch
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return x_batch, y_batch

    # Membuat langkah kerja bernama `on_epoch_end`.
    def on_epoch_end(self) -> None:
        # Mengecek syarat sebelum melanjutkan proses.
        if self.shuffle:
            # Melanjutkan langkah kerja pada bagian kode ini.
            self._rng.shuffle(self.entries)


# Membuat langkah kerja bernama `collect_targets`.
def collect_targets(sequence: NativePatchSequence) -> np.ndarray:
    # Menyimpan nilai ke `targets` untuk dipakai pada langkah berikutnya.
    targets: list[np.ndarray] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for batch_index in range(len(sequence)):
        # Menyimpan nilai ke `_, y_batch` untuk dipakai pada langkah berikutnya.
        _, y_batch = sequence[batch_index]
        # Melanjutkan langkah kerja pada bagian kode ini.
        targets.append(y_batch)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.concatenate(targets, axis=0)


# Membuat langkah kerja bernama `compute_metrics`.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, ground_truth_threshold: float) -> dict[str, float]:
    # Menyimpan nilai ke `truth` untuk dipakai pada langkah berikutnya.
    truth = y_true >= ground_truth_threshold
    # Menyimpan nilai ke `pred` untuk dipakai pada langkah berikutnya.
    pred = y_pred >= threshold
    # Menyimpan nilai ke `tp` untuk dipakai pada langkah berikutnya.
    tp = int(np.logical_and(pred, truth).sum())
    # Menyimpan nilai ke `fp` untuk dipakai pada langkah berikutnya.
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    # Menyimpan nilai ke `fn` untuk dipakai pada langkah berikutnya.
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    # Menyimpan nilai ke `tn` untuk dipakai pada langkah berikutnya.
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())

    # Membuat langkah kerja bernama `safe_div`.
    def safe_div(a: float, b: float) -> float:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return 0.0 if b == 0 else float(a / b)

    # Menyimpan nilai ke `precision` untuk dipakai pada langkah berikutnya.
    precision = safe_div(tp, tp + fp)
    # Menyimpan nilai ke `recall` untuk dipakai pada langkah berikutnya.
    recall = safe_div(tp, tp + fn)
    # Menyimpan nilai ke `f1_score` untuk dipakai pada langkah berikutnya.
    f1_score = safe_div(2 * precision * recall, precision + recall)
    # Menyimpan nilai ke `iou` untuk dipakai pada langkah berikutnya.
    iou = safe_div(tp, tp + fp + fn)
    # Menyimpan nilai ke `accuracy` untuk dipakai pada langkah berikutnya.
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tp": tp,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "fp": fp,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "fn": fn,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tn": tn,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "precision": precision,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recall": recall,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "f1_score": f1_score,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "iou": iou,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "accuracy": accuracy,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }


# Membuat langkah kerja bernama `_dilate_binary_array`.
def _dilate_binary_array(mask: np.ndarray, buffer_radius: int) -> np.ndarray:
    # Mengecek syarat sebelum melanjutkan proses.
    if buffer_radius <= 0:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return mask.astype(bool)

    # Menyimpan nilai ke `original_shape` untuk dipakai pada langkah berikutnya.
    original_shape = mask.shape
    # Menyimpan nilai ke `squeezed` untuk dipakai pada langkah berikutnya.
    squeezed = np.squeeze(mask.astype(np.uint8), axis=-1) if mask.ndim == 4 and mask.shape[-1] == 1 else mask.astype(np.uint8)
    # Mengecek syarat sebelum melanjutkan proses.
    if squeezed.ndim == 2:
        # Menyimpan nilai ke `flat` untuk dipakai pada langkah berikutnya.
        flat = squeezed.reshape(1, squeezed.shape[0], squeezed.shape[1])
    # Mengecek syarat lain jika syarat sebelumnya tidak terpenuhi.
    elif squeezed.ndim == 3:
        # Menyimpan nilai ke `flat` untuk dipakai pada langkah berikutnya.
        flat = squeezed.reshape(-1, squeezed.shape[-2], squeezed.shape[-1])
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError(f"Shape mask tidak didukung untuk buffer evaluation: {mask.shape}")

    # Menyimpan nilai ke `kernel_size` untuk dipakai pada langkah berikutnya.
    kernel_size = _normalize_kernel((int(buffer_radius) * 2) + 1)
    # Menyimpan nilai ke `dilated` untuk dipakai pada langkah berikutnya.
    dilated: list[np.ndarray] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for item in flat:
        # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
        image = Image.fromarray((item > 0).astype(np.uint8) * 255)
        # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
        image = image.filter(ImageFilter.MaxFilter(size=kernel_size))
        # Menyimpan nilai ke `dilated.append(np.asarray(image, dtype` untuk dipakai pada langkah berikutnya.
        dilated.append(np.asarray(image, dtype=np.uint8) > 0)

    # Menyimpan nilai ke `stacked` untuk dipakai pada langkah berikutnya.
    stacked = np.stack(dilated, axis=0)
    # Mengecek syarat sebelum melanjutkan proses.
    if squeezed.ndim == 2:
        # Menyimpan nilai ke `result` untuk dipakai pada langkah berikutnya.
        result = stacked[0]
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menyimpan nilai ke `result` untuk dipakai pada langkah berikutnya.
        result = stacked.reshape(squeezed.shape)

    # Mengecek syarat sebelum melanjutkan proses.
    if len(original_shape) == 4 and original_shape[-1] == 1:
        # Menyimpan nilai ke `result` untuk dipakai pada langkah berikutnya.
        result = np.expand_dims(result, axis=-1)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return result.astype(bool)


# Membuat langkah kerja bernama `compute_buffered_metrics`.
def compute_buffered_metrics(
    # Menjelaskan data `y_true` yang disimpan atau dikirim pada bagian ini.
    y_true: np.ndarray,
    # Menjelaskan data `y_pred` yang disimpan atau dikirim pada bagian ini.
    y_pred: np.ndarray,
    # Menjelaskan data `threshold` yang disimpan atau dikirim pada bagian ini.
    threshold: float,
    # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
    ground_truth_threshold: float,
    # Menjelaskan data `buffer_radius` yang disimpan atau dikirim pada bagian ini.
    buffer_radius: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> dict[str, float]:
    # Menyimpan nilai ke `truth` untuk dipakai pada langkah berikutnya.
    truth = y_true >= ground_truth_threshold
    # Menyimpan nilai ke `pred` untuk dipakai pada langkah berikutnya.
    pred = y_pred >= threshold
    # Menyimpan nilai ke `truth` untuk dipakai pada langkah berikutnya.
    truth = _dilate_binary_array(truth, buffer_radius)
    # Menyimpan nilai ke `pred` untuk dipakai pada langkah berikutnya.
    pred = _dilate_binary_array(pred, buffer_radius)

    # Menyimpan nilai ke `tp` untuk dipakai pada langkah berikutnya.
    tp = int(np.logical_and(pred, truth).sum())
    # Menyimpan nilai ke `fp` untuk dipakai pada langkah berikutnya.
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    # Menyimpan nilai ke `fn` untuk dipakai pada langkah berikutnya.
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    # Menyimpan nilai ke `tn` untuk dipakai pada langkah berikutnya.
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())

    # Membuat langkah kerja bernama `safe_div`.
    def safe_div(a: float, b: float) -> float:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return 0.0 if b == 0 else float(a / b)

    # Menyimpan nilai ke `precision` untuk dipakai pada langkah berikutnya.
    precision = safe_div(tp, tp + fp)
    # Menyimpan nilai ke `recall` untuk dipakai pada langkah berikutnya.
    recall = safe_div(tp, tp + fn)
    # Menyimpan nilai ke `f1_score` untuk dipakai pada langkah berikutnya.
    f1_score = safe_div(2 * precision * recall, precision + recall)
    # Menyimpan nilai ke `iou` untuk dipakai pada langkah berikutnya.
    iou = safe_div(tp, tp + fp + fn)
    # Menyimpan nilai ke `accuracy` untuk dipakai pada langkah berikutnya.
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "buffer_radius": int(buffer_radius),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tp": tp,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "fp": fp,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "fn": fn,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tn": tn,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "precision": precision,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recall": recall,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "f1_score": f1_score,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "iou": iou,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "accuracy": accuracy,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }


# Membuat langkah kerja bernama `_threshold_grid`.
def _threshold_grid(step: float, *, start: float = 0.05, stop: float = 0.95) -> list[float]:
    # Menyimpan nilai ke `step` untuk dipakai pada langkah berikutnya.
    step = max(float(step), 0.001)
    # Menyimpan nilai ke `start` untuk dipakai pada langkah berikutnya.
    start = max(0.01, float(start))
    # Menyimpan nilai ke `stop` untuk dipakai pada langkah berikutnya.
    stop = min(0.99, float(stop))
    # Mengecek syarat sebelum melanjutkan proses.
    if stop < start:
        # Menyimpan nilai ke `start, stop` untuk dipakai pada langkah berikutnya.
        start, stop = stop, start
    # Menyimpan nilai ke `values` untuk dipakai pada langkah berikutnya.
    values = np.arange(start, stop + (step * 0.5), step)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return sorted({round(float(np.clip(value, 0.01, 0.99)), 4) for value in values})


# Membuat langkah kerja bernama `_metrics_rank`.
def _metrics_rank(metrics: dict[str, float]) -> tuple[float, float, float, float, int]:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return (
        # Melanjutkan langkah kerja pada bagian kode ini.
        metrics["f1_score"],
        # Melanjutkan langkah kerja pada bagian kode ini.
        metrics["iou"],
        # Melanjutkan langkah kerja pada bagian kode ini.
        metrics["precision"],
        # Melanjutkan langkah kerja pada bagian kode ini.
        metrics["recall"],
        # Melanjutkan langkah kerja pada bagian kode ini.
        -metrics["fp"],
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )


# Membuat langkah kerja bernama `sweep_thresholds`.
def sweep_thresholds(
    # Menjelaskan data `y_true` yang disimpan atau dikirim pada bagian ini.
    y_true: np.ndarray,
    # Menjelaskan data `y_pred` yang disimpan atau dikirim pada bagian ini.
    y_pred: np.ndarray,
    # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
    ground_truth_threshold: float,
    # Menjelaskan data `sweep_step` yang disimpan atau dikirim pada bagian ini.
    sweep_step: float,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    # Menyimpan nilai ke `evaluated` untuk dipakai pada langkah berikutnya.
    evaluated: dict[float, dict[str, float]] = {}
    # Menyimpan nilai ke `best_threshold` untuk dipakai pada langkah berikutnya.
    best_threshold = 0.5
    # Menyimpan nilai ke `best_metrics` untuk dipakai pada langkah berikutnya.
    best_metrics: dict[str, float] | None = None

    # Membuat langkah kerja bernama `evaluate_thresholds`.
    def evaluate_thresholds(thresholds: list[float]) -> None:
        # Melanjutkan langkah kerja pada bagian kode ini.
        nonlocal best_threshold, best_metrics
        # Mengulang proses untuk setiap data dalam daftar.
        for threshold in thresholds:
            # Mengecek syarat sebelum melanjutkan proses.
            if threshold not in evaluated:
                # Menyimpan nilai ke `evaluated[threshold]` untuk dipakai pada langkah berikutnya.
                evaluated[threshold] = compute_metrics(y_true, y_pred, threshold, ground_truth_threshold)
            # Menyimpan nilai ke `metrics` untuk dipakai pada langkah berikutnya.
            metrics = evaluated[threshold]
            # Mengecek syarat sebelum melanjutkan proses.
            if best_metrics is None or _metrics_rank(metrics) > _metrics_rank(best_metrics):
                # Menyimpan nilai ke `best_threshold` untuk dipakai pada langkah berikutnya.
                best_threshold = threshold
                # Menyimpan nilai ke `best_metrics` untuk dipakai pada langkah berikutnya.
                best_metrics = metrics

    # Melanjutkan langkah kerja pada bagian kode ini.
    evaluate_thresholds(_threshold_grid(sweep_step))

    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert best_metrics is not None
    # Menyimpan nilai ke `results` untuk dipakai pada langkah berikutnya.
    results = [{"threshold": threshold, **evaluated[threshold]} for threshold in sorted(evaluated)]
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return best_threshold, results, best_metrics


# Membuat wadah bernama `ValidationSweepCallback` untuk menyimpan data atau aturan kerja.
class ValidationSweepCallback(tf.keras.callbacks.Callback):
    # Membuat langkah kerja bernama `__init__`.
    def __init__(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        self,
        # Menjelaskan data `val_sequence` yang disimpan atau dikirim pada bagian ini.
        val_sequence: NativePatchSequence,
        # Menjelaskan data `y_val` yang disimpan atau dikirim pada bagian ini.
        y_val: np.ndarray,
        # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
        ground_truth_threshold: float,
        # Menjelaskan data `sweep_step` yang disimpan atau dikirim pada bagian ini.
        sweep_step: float,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ) -> None:
        # Melanjutkan langkah kerja pada bagian kode ini.
        super().__init__()
        # Menyimpan nilai ke `self.val_sequence` untuk dipakai pada langkah berikutnya.
        self.val_sequence = val_sequence
        # Menyimpan nilai ke `self.y_val` untuk dipakai pada langkah berikutnya.
        self.y_val = y_val
        # Menyimpan nilai ke `self.ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        self.ground_truth_threshold = ground_truth_threshold
        # Menyimpan nilai ke `self.sweep_step` untuk dipakai pada langkah berikutnya.
        self.sweep_step = sweep_step

    # Membuat langkah kerja bernama `on_epoch_end`.
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        # Menyimpan nilai ke `logs` untuk dipakai pada langkah berikutnya.
        logs = logs or {}
        # Menyimpan nilai ke `predictions` untuk dipakai pada langkah berikutnya.
        predictions = self.model.predict(self.val_sequence, verbose=0)
        # Menyimpan nilai ke `best_threshold, _, best_metrics` untuk dipakai pada langkah berikutnya.
        best_threshold, _, best_metrics = sweep_thresholds(
            # Melanjutkan langkah kerja pada bagian kode ini.
            self.y_val,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            predictions,
            # Melanjutkan langkah kerja pada bagian kode ini.
            self.ground_truth_threshold,
            # Melanjutkan langkah kerja pada bagian kode ini.
            self.sweep_step,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `logs["val_best_threshold"]` untuk dipakai pada langkah berikutnya.
        logs["val_best_threshold"] = float(best_threshold)
        # Menyimpan nilai ke `logs["val_best_precision"]` untuk dipakai pada langkah berikutnya.
        logs["val_best_precision"] = float(best_metrics["precision"])
        # Menyimpan nilai ke `logs["val_best_recall"]` untuk dipakai pada langkah berikutnya.
        logs["val_best_recall"] = float(best_metrics["recall"])
        # Menyimpan nilai ke `logs["val_best_f1"]` untuk dipakai pada langkah berikutnya.
        logs["val_best_f1"] = float(best_metrics["f1_score"])
        # Menyimpan nilai ke `logs["val_best_iou"]` untuk dipakai pada langkah berikutnya.
        logs["val_best_iou"] = float(best_metrics["iou"])
        # Menyimpan nilai ke `logs["val_model_score"]` untuk dipakai pada langkah berikutnya.
        logs["val_model_score"] = float(self._compute_model_score(best_metrics))
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(
            # Menyimpan nilai ke `f"\nEpoch {epoch + 1}` untuk dipakai pada langkah berikutnya.
            f"\nEpoch {epoch + 1}: val_best_threshold={best_threshold:.2f} "
            # Menyimpan nilai ke `f"val_best_f1` untuk dipakai pada langkah berikutnya.
            f"val_best_f1={best_metrics['f1_score']:.4f} "
            # Menyimpan nilai ke `f"val_best_iou` untuk dipakai pada langkah berikutnya.
            f"val_best_iou={best_metrics['iou']:.4f} "
            # Menyimpan nilai ke `f"val_best_recall` untuk dipakai pada langkah berikutnya.
            f"val_best_recall={best_metrics['recall']:.4f}"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
    @staticmethod
    # Membuat langkah kerja bernama `_compute_model_score`.
    def _compute_model_score(metrics: dict[str, float]) -> float:
        # Mengecek syarat sebelum melanjutkan proses.
        if metrics["precision"] < 0.02 or metrics["f1_score"] < 0.02:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return 0.0
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return float(
            # Melanjutkan langkah kerja pada bagian kode ini.
            (0.60 * metrics["f1_score"])
            # Melanjutkan langkah kerja pada bagian kode ini.
            + (0.20 * metrics["iou"])
            # Melanjutkan langkah kerja pada bagian kode ini.
            + (0.10 * metrics["precision"])
            # Melanjutkan langkah kerja pada bagian kode ini.
            + (0.10 * metrics["recall"])
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )


# Membuat langkah kerja bernama `build_patch_entries`.
def build_patch_entries(
    # Menjelaskan data `records` yang disimpan atau dikirim pada bagian ini.
    records: list[DatasetRecord],
    # Menjelaskan data `sample_starts` yang disimpan atau dikirim pada bagian ini.
    sample_starts: list[int],
    # Melanjutkan langkah kerja pada bagian kode ini.
    *,
    # Menjelaskan data `seq_length` yang disimpan atau dikirim pada bagian ini.
    seq_length: int,
    # Menjelaskan data `native_width` yang disimpan atau dikirim pada bagian ini.
    native_width: int,
    # Menjelaskan data `native_height` yang disimpan atau dikirim pada bagian ini.
    native_height: int,
    # Menjelaskan data `label_dilation_kernel` yang disimpan atau dikirim pada bagian ini.
    label_dilation_kernel: int,
    # Menjelaskan data `label_blur_radius` yang disimpan atau dikirim pada bagian ini.
    label_blur_radius: float,
    # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
    ground_truth_threshold: float,
    # Menjelaskan data `positive_patches` yang disimpan atau dikirim pada bagian ini.
    positive_patches: int,
    # Menjelaskan data `negative_patches` yang disimpan atau dikirim pada bagian ini.
    negative_patches: int,
    # Menjelaskan data `seed` yang disimpan atau dikirim pada bagian ini.
    seed: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    # Menyimpan nilai ke `rng` untuk dipakai pada langkah berikutnya.
    rng = np.random.default_rng(seed)
    # Menyimpan nilai ke `entries` untuk dipakai pada langkah berikutnya.
    entries: list[tuple[int, int, int]] = []
    # Menyimpan nilai ke `positive_samples` untuk dipakai pada langkah berikutnya.
    positive_samples = 0
    # Menyimpan nilai ke `negative_samples` untuk dipakai pada langkah berikutnya.
    negative_samples = 0

    # Mengulang proses untuk setiap data dalam daftar.
    for start in sample_starts:
        # Menyimpan nilai ke `target_mask` untuk dipakai pada langkah berikutnya.
        target_mask = load_native_risk_map(
            # Melanjutkan langkah kerja pada bagian kode ini.
            str(records[start + seq_length].path),
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            native_width,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            native_height,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            label_dilation_kernel,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            label_blur_radius,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `has_positive` untuk dipakai pada langkah berikutnya.
        has_positive = bool(np.any(target_mask >= ground_truth_threshold))
        # Mengecek syarat sebelum melanjutkan proses.
        if has_positive:
            # Menyimpan nilai ke `positive_samples +` untuk dipakai pada langkah berikutnya.
            positive_samples += 1
        # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
        else:
            # Menyimpan nilai ke `negative_samples +` untuk dipakai pada langkah berikutnya.
            negative_samples += 1

        # Menyimpan nilai ke `centers` untuk dipakai pada langkah berikutnya.
        centers = sample_patch_centers(
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            target_mask,
            # Menyimpan nilai ke `positive_patch_count` untuk dipakai pada langkah berikutnya.
            positive_patch_count=positive_patches if has_positive else 0,
            # Menyimpan nilai ke `negative_patch_count` untuk dipakai pada langkah berikutnya.
            negative_patch_count=negative_patches,
            # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
            ground_truth_threshold=ground_truth_threshold,
            # Menyimpan nilai ke `rng` untuk dipakai pada langkah berikutnya.
            rng=rng,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Melanjutkan langkah kerja pada bagian kode ini.
        entries.extend((start, cy, cx) for cy, cx in centers)

    # Menyimpan nilai ke `stats` untuk dipakai pada langkah berikutnya.
    stats = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sample_count": len(sample_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_sample_count": positive_samples,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "negative_sample_count": negative_samples,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_entry_count": len(entries),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_patches_per_sample": positive_patches,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "negative_patches_per_sample": negative_patches,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return entries, stats


# Membuat langkah kerja bernama `run_training`.
def run_training(args: argparse.Namespace, output_model: Path, output_report: Path) -> dict:
    # Menyimpan nilai ke `dataset_dir` untuk dipakai pada langkah berikutnya.
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    # Menyimpan nilai ke `image_extensions` untuk dipakai pada langkah berikutnya.
    image_extensions = parse_image_extensions(args.image_extensions)
    # Menyimpan nilai ke `records` untuk dipakai pada langkah berikutnya.
    records = load_records(dataset_dir, image_extensions)
    # Mengecek syarat sebelum melanjutkan proses.
    if len(records) <= args.seq_length:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Jumlah data tidak cukup untuk membentuk sequence.")

    # Menyimpan nilai ke `records, skipped_records` untuk dipakai pada langkah berikutnya.
    records, skipped_records = validate_records(records)
    # Mengecek syarat sebelum melanjutkan proses.
    if not records:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Tidak ada file gambar valid di dataset. Periksa --dataset-dir dan --image-extensions.")
    # Mengecek syarat sebelum melanjutkan proses.
    if skipped_records:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"File dataset yang dilewati: {len(skipped_records)}")
        # Mengulang proses untuk setiap data dalam daftar.
        for item in skipped_records[:10]:
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print(f"- skip {item['path']} | alasan: {item['reason']}")
    # Mengecek syarat sebelum melanjutkan proses.
    if len(records) <= args.seq_length:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Jumlah data valid tidak cukup untuk membentuk sequence.")

    # Menyimpan nilai ke `sample_starts` untuk dipakai pada langkah berikutnya.
    sample_starts = build_sample_starts(len(records), args.seq_length)
    # Menyimpan nilai ke `train_starts, val_starts, test_starts` untuk dipakai pada langkah berikutnya.
    train_starts, val_starts, test_starts = split_sample_starts(sample_starts, args.train_ratio, args.val_ratio)
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Dataset ditemukan: {len(records)} frame")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Rentang data: {records[0].date.date()} s.d. {records[-1].date.date()}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Train samples: {len(train_starts)}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Val samples: {len(val_starts)}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Test samples: {len(test_starts)}")

    # Menyimpan nilai ke `pos_weight, positive_ratio` untuk dipakai pada langkah berikutnya.
    pos_weight, positive_ratio = estimate_positive_weight(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        records,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        train_starts,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.seq_length,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.native_width,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.native_height,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.label_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.label_blur_radius,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.max_pos_weight,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Positive mass ratio (train): {positive_ratio:.8f}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Positive class weight: {pos_weight:.4f}")

    # Menyimpan nilai ke `train_entries, train_patch_stats` untuk dipakai pada langkah berikutnya.
    train_entries, train_patch_stats = build_patch_entries(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        records,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        train_starts,
        # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
        seq_length=args.seq_length,
        # Menyimpan nilai ke `native_width` untuk dipakai pada langkah berikutnya.
        native_width=args.native_width,
        # Menyimpan nilai ke `native_height` untuk dipakai pada langkah berikutnya.
        native_height=args.native_height,
        # Menyimpan nilai ke `label_dilation_kernel` untuk dipakai pada langkah berikutnya.
        label_dilation_kernel=args.label_dilation_kernel,
        # Menyimpan nilai ke `label_blur_radius` untuk dipakai pada langkah berikutnya.
        label_blur_radius=args.label_blur_radius,
        # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        ground_truth_threshold=args.ground_truth_threshold,
        # Menyimpan nilai ke `positive_patches` untuk dipakai pada langkah berikutnya.
        positive_patches=args.train_positive_patches,
        # Menyimpan nilai ke `negative_patches` untuk dipakai pada langkah berikutnya.
        negative_patches=args.train_negative_patches,
        # Menyimpan nilai ke `seed` untuk dipakai pada langkah berikutnya.
        seed=args.seed,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `val_entries, val_patch_stats` untuk dipakai pada langkah berikutnya.
    val_entries, val_patch_stats = build_patch_entries(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        records,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        val_starts,
        # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
        seq_length=args.seq_length,
        # Menyimpan nilai ke `native_width` untuk dipakai pada langkah berikutnya.
        native_width=args.native_width,
        # Menyimpan nilai ke `native_height` untuk dipakai pada langkah berikutnya.
        native_height=args.native_height,
        # Menyimpan nilai ke `label_dilation_kernel` untuk dipakai pada langkah berikutnya.
        label_dilation_kernel=args.label_dilation_kernel,
        # Menyimpan nilai ke `label_blur_radius` untuk dipakai pada langkah berikutnya.
        label_blur_radius=args.label_blur_radius,
        # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        ground_truth_threshold=args.ground_truth_threshold,
        # Menyimpan nilai ke `positive_patches` untuk dipakai pada langkah berikutnya.
        positive_patches=args.eval_positive_patches,
        # Menyimpan nilai ke `negative_patches` untuk dipakai pada langkah berikutnya.
        negative_patches=args.eval_negative_patches,
        # Menyimpan nilai ke `seed` untuk dipakai pada langkah berikutnya.
        seed=args.seed + 1,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `test_entries, test_patch_stats` untuk dipakai pada langkah berikutnya.
    test_entries, test_patch_stats = build_patch_entries(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        records,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        test_starts,
        # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
        seq_length=args.seq_length,
        # Menyimpan nilai ke `native_width` untuk dipakai pada langkah berikutnya.
        native_width=args.native_width,
        # Menyimpan nilai ke `native_height` untuk dipakai pada langkah berikutnya.
        native_height=args.native_height,
        # Menyimpan nilai ke `label_dilation_kernel` untuk dipakai pada langkah berikutnya.
        label_dilation_kernel=args.label_dilation_kernel,
        # Menyimpan nilai ke `label_blur_radius` untuk dipakai pada langkah berikutnya.
        label_blur_radius=args.label_blur_radius,
        # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        ground_truth_threshold=args.ground_truth_threshold,
        # Menyimpan nilai ke `positive_patches` untuk dipakai pada langkah berikutnya.
        positive_patches=args.eval_positive_patches,
        # Menyimpan nilai ke `negative_patches` untuk dipakai pada langkah berikutnya.
        negative_patches=args.eval_negative_patches,
        # Menyimpan nilai ke `seed` untuk dipakai pada langkah berikutnya.
        seed=args.seed + 2,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Train patch entries: {len(train_entries)}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Val patch entries: {len(val_entries)}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Test patch entries: {len(test_entries)}")

    # Menyimpan nilai ke `train_sequence` untuk dipakai pada langkah berikutnya.
    train_sequence = NativePatchSequence(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        records,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        train_entries,
        # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
        seq_length=args.seq_length,
        # Menyimpan nilai ke `native_width` untuk dipakai pada langkah berikutnya.
        native_width=args.native_width,
        # Menyimpan nilai ke `native_height` untuk dipakai pada langkah berikutnya.
        native_height=args.native_height,
        # Menyimpan nilai ke `patch_width` untuk dipakai pada langkah berikutnya.
        patch_width=args.patch_width,
        # Menyimpan nilai ke `patch_height` untuk dipakai pada langkah berikutnya.
        patch_height=args.patch_height,
        # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
        input_dilation_kernel=args.input_dilation_kernel,
        # Menyimpan nilai ke `label_dilation_kernel` untuk dipakai pada langkah berikutnya.
        label_dilation_kernel=args.label_dilation_kernel,
        # Menyimpan nilai ke `label_blur_radius` untuk dipakai pada langkah berikutnya.
        label_blur_radius=args.label_blur_radius,
        # Menyimpan nilai ke `feature_stack` untuk dipakai pada langkah berikutnya.
        feature_stack=args.feature_stack,
        # Menyimpan nilai ke `context_kernel` untuk dipakai pada langkah berikutnya.
        context_kernel=args.context_kernel,
        # Menyimpan nilai ke `batch_size` untuk dipakai pada langkah berikutnya.
        batch_size=args.batch_size,
        # Menyimpan nilai ke `shuffle` untuk dipakai pada langkah berikutnya.
        shuffle=True,
        # Menyimpan nilai ke `enable_augmentation` untuk dipakai pada langkah berikutnya.
        enable_augmentation=args.enable_augmentation,
        # Menyimpan nilai ke `seed` untuk dipakai pada langkah berikutnya.
        seed=args.seed,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `val_sequence` untuk dipakai pada langkah berikutnya.
    val_sequence = NativePatchSequence(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        records,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        val_entries,
        # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
        seq_length=args.seq_length,
        # Menyimpan nilai ke `native_width` untuk dipakai pada langkah berikutnya.
        native_width=args.native_width,
        # Menyimpan nilai ke `native_height` untuk dipakai pada langkah berikutnya.
        native_height=args.native_height,
        # Menyimpan nilai ke `patch_width` untuk dipakai pada langkah berikutnya.
        patch_width=args.patch_width,
        # Menyimpan nilai ke `patch_height` untuk dipakai pada langkah berikutnya.
        patch_height=args.patch_height,
        # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
        input_dilation_kernel=args.input_dilation_kernel,
        # Menyimpan nilai ke `label_dilation_kernel` untuk dipakai pada langkah berikutnya.
        label_dilation_kernel=args.label_dilation_kernel,
        # Menyimpan nilai ke `label_blur_radius` untuk dipakai pada langkah berikutnya.
        label_blur_radius=args.label_blur_radius,
        # Menyimpan nilai ke `feature_stack` untuk dipakai pada langkah berikutnya.
        feature_stack=args.feature_stack,
        # Menyimpan nilai ke `context_kernel` untuk dipakai pada langkah berikutnya.
        context_kernel=args.context_kernel,
        # Menyimpan nilai ke `batch_size` untuk dipakai pada langkah berikutnya.
        batch_size=args.batch_size,
        # Menyimpan nilai ke `shuffle` untuk dipakai pada langkah berikutnya.
        shuffle=False,
        # Menyimpan nilai ke `enable_augmentation` untuk dipakai pada langkah berikutnya.
        enable_augmentation=False,
        # Menyimpan nilai ke `seed` untuk dipakai pada langkah berikutnya.
        seed=args.seed + 10,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `test_sequence` untuk dipakai pada langkah berikutnya.
    test_sequence = NativePatchSequence(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        records,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        test_entries,
        # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
        seq_length=args.seq_length,
        # Menyimpan nilai ke `native_width` untuk dipakai pada langkah berikutnya.
        native_width=args.native_width,
        # Menyimpan nilai ke `native_height` untuk dipakai pada langkah berikutnya.
        native_height=args.native_height,
        # Menyimpan nilai ke `patch_width` untuk dipakai pada langkah berikutnya.
        patch_width=args.patch_width,
        # Menyimpan nilai ke `patch_height` untuk dipakai pada langkah berikutnya.
        patch_height=args.patch_height,
        # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
        input_dilation_kernel=args.input_dilation_kernel,
        # Menyimpan nilai ke `label_dilation_kernel` untuk dipakai pada langkah berikutnya.
        label_dilation_kernel=args.label_dilation_kernel,
        # Menyimpan nilai ke `label_blur_radius` untuk dipakai pada langkah berikutnya.
        label_blur_radius=args.label_blur_radius,
        # Menyimpan nilai ke `feature_stack` untuk dipakai pada langkah berikutnya.
        feature_stack=args.feature_stack,
        # Menyimpan nilai ke `context_kernel` untuk dipakai pada langkah berikutnya.
        context_kernel=args.context_kernel,
        # Menyimpan nilai ke `batch_size` untuk dipakai pada langkah berikutnya.
        batch_size=args.batch_size,
        # Menyimpan nilai ke `shuffle` untuk dipakai pada langkah berikutnya.
        shuffle=False,
        # Menyimpan nilai ke `enable_augmentation` untuk dipakai pada langkah berikutnya.
        enable_augmentation=False,
        # Menyimpan nilai ke `seed` untuk dipakai pada langkah berikutnya.
        seed=args.seed + 20,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `y_val` untuk dipakai pada langkah berikutnya.
    y_val = collect_targets(val_sequence)
    # Menyimpan nilai ke `y_test` untuk dipakai pada langkah berikutnya.
    y_test = collect_targets(test_sequence)
    # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
    model = build_model(
        # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
        seq_length=args.seq_length,
        # Menyimpan nilai ke `patch_height` untuk dipakai pada langkah berikutnya.
        patch_height=args.patch_height,
        # Menyimpan nilai ke `patch_width` untuk dipakai pada langkah berikutnya.
        patch_width=args.patch_width,
        # Menyimpan nilai ke `channels` untuk dipakai pada langkah berikutnya.
        channels=feature_stack_channels(args.feature_stack),
        # Menyimpan nilai ke `pos_weight` untuk dipakai pada langkah berikutnya.
        pos_weight=pos_weight,
        # Menyimpan nilai ke `loss_strategy` untuk dipakai pada langkah berikutnya.
        loss_strategy=args.loss_strategy,
        # Menyimpan nilai ke `context_kernel` untuk dipakai pada langkah berikutnya.
        context_kernel=args.context_kernel,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `output_model.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_model.parent.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `output_report.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_report.parent.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `callbacks` untuk dipakai pada langkah berikutnya.
    callbacks = [
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        ValidationSweepCallback(
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            val_sequence,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            y_val,
            # Melanjutkan langkah kerja pada bagian kode ini.
            args.ground_truth_threshold,
            # Melanjutkan langkah kerja pada bagian kode ini.
            args.threshold_sweep_step,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
        # Menyimpan nilai ke `ModelCheckpoint(str(output_model), monitor` untuk dipakai pada langkah berikutnya.
        ModelCheckpoint(str(output_model), monitor="val_model_score", save_best_only=True, mode="max", verbose=1),
        # Menyimpan nilai ke `EarlyStopping(monitor` untuk dipakai pada langkah berikutnya.
        EarlyStopping(monitor="val_model_score", patience=8, restore_best_weights=True, mode="max", verbose=1),
        # Menyimpan nilai ke `ReduceLROnPlateau(monitor` untuk dipakai pada langkah berikutnya.
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ]

    # Menyimpan nilai ke `history` untuk dipakai pada langkah berikutnya.
    history = model.fit(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        train_sequence,
        # Menyimpan nilai ke `validation_data` untuk dipakai pada langkah berikutnya.
        validation_data=val_sequence,
        # Menyimpan nilai ke `epochs` untuk dipakai pada langkah berikutnya.
        epochs=args.epochs,
        # Menyimpan nilai ke `callbacks` untuk dipakai pada langkah berikutnya.
        callbacks=callbacks,
        # Menyimpan nilai ke `verbose` untuk dipakai pada langkah berikutnya.
        verbose=1,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `val_predictions` untuk dipakai pada langkah berikutnya.
    val_predictions = model.predict(val_sequence, verbose=0)
    # Menyimpan nilai ke `best_threshold, val_sweep, best_val_metrics` untuk dipakai pada langkah berikutnya.
    best_threshold, val_sweep, best_val_metrics = sweep_thresholds(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_val,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        val_predictions,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.ground_truth_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.threshold_sweep_step,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `test_predictions` untuk dipakai pada langkah berikutnya.
    test_predictions = model.predict(test_sequence, verbose=0)
    # Menyimpan nilai ke `default_test_metrics` untuk dipakai pada langkah berikutnya.
    default_test_metrics = compute_metrics(y_test, test_predictions, args.threshold, args.ground_truth_threshold)
    # Menyimpan nilai ke `best_test_metrics` untuk dipakai pada langkah berikutnya.
    best_test_metrics = compute_metrics(y_test, test_predictions, best_threshold, args.ground_truth_threshold)
    # Menyimpan nilai ke `default_test_buffered_metrics` untuk dipakai pada langkah berikutnya.
    default_test_buffered_metrics = compute_buffered_metrics(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_test,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        test_predictions,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.ground_truth_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.evaluation_buffer_radius,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `best_test_buffered_metrics` untuk dipakai pada langkah berikutnya.
    best_test_buffered_metrics = compute_buffered_metrics(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_test,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        test_predictions,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        best_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.ground_truth_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.evaluation_buffer_radius,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `best_val_buffered_metrics` untuk dipakai pada langkah berikutnya.
    best_val_buffered_metrics = compute_buffered_metrics(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_val,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        val_predictions,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        best_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.ground_truth_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.evaluation_buffer_radius,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `report` untuk dipakai pada langkah berikutnya.
    report = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "mode": "historical_risk_patch_1528x773",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dataset_dir": str(dataset_dir),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "image_extensions": list(image_extensions),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dataset_frame_count": len(records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "skipped_frame_count": len(skipped_records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "skipped_records_preview": skipped_records[:20],
        # Melanjutkan langkah kerja pada bagian kode ini.
        "date_start": records[0].date.strftime("%Y-%m-%d"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "date_end": records[-1].date.strftime("%Y-%m-%d"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "native_width": args.native_width,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "native_height": args.native_height,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_width": args.patch_width,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_height": args.patch_height,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "seq_length": args.seq_length,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "channels": feature_stack_channels(args.feature_stack),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "feature_stack": args.feature_stack,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "epochs_requested": args.epochs,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "batch_size": args.batch_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_ratio": args.train_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_ratio": args.val_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "default_threshold": args.threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "ground_truth_threshold": args.ground_truth_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "selected_threshold": best_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "threshold_sweep_step": args.threshold_sweep_step,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "legacy_dilation_kernel": args.dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_dilation_kernel": args.input_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "label_dilation_kernel": args.label_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "label_blur_radius": args.label_blur_radius,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "evaluation_buffer_radius": args.evaluation_buffer_radius,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "context_kernel": args.context_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "loss_strategy": args.loss_strategy,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_augmentation_enabled": args.enable_augmentation,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_ratio_train": positive_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_class_weight": pos_weight,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_samples": len(train_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_samples": len(val_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_samples": len(test_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_patch_stats": train_patch_stats,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_patch_stats": val_patch_stats,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_patch_stats": test_patch_stats,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "history": history.history,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "validation_threshold_sweep": val_sweep,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "validation_model_score": ValidationSweepCallback._compute_model_score(best_val_metrics),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "validation_best_metrics": best_val_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "validation_buffered_metrics_best_threshold": best_val_buffered_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_metrics_default_threshold": default_test_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_metrics_best_threshold": best_test_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_buffered_metrics_default_threshold": default_test_buffered_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_buffered_metrics_best_threshold": best_test_buffered_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_metrics": best_test_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "output_model": str(output_model),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Menyimpan nilai ke `output_report.write_text(json.dumps(report, indent` untuk dipakai pada langkah berikutnya.
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(json.dumps(report["validation_best_metrics"], indent=2))
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(json.dumps(report["test_metrics_best_threshold"], indent=2))
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(json.dumps(report["test_buffered_metrics_best_threshold"], indent=2))
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Threshold terbaik dari validation: {best_threshold}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Model terbaik tersimpan di: {output_model}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[historical_risk_patch] Report training tersimpan di: {output_report}")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return report


# Membuat langkah kerja bernama `main`.
def main() -> int:
    # Melanjutkan langkah kerja pada bagian kode ini.
    maybe_mount_drive()
    # Melanjutkan langkah kerja pada bagian kode ini.
    configure_tensorflow_runtime()
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parse_args()
    # Menyimpan nilai ke `output_model, output_report` untuk dipakai pada langkah berikutnya.
    output_model, output_report = resolve_output_paths(
        # Menyimpan nilai ke `output_model` untuk dipakai pada langkah berikutnya.
        output_model=Path(args.output_model).resolve(),
        # Menyimpan nilai ke `output_report` untuk dipakai pada langkah berikutnya.
        output_report=Path(args.output_report).resolve(),
        # Menyimpan nilai ke `overwrite` untuk dipakai pada langkah berikutnya.
        overwrite=args.overwrite,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Melanjutkan langkah kerja pada bagian kode ini.
    run_training(args, output_model, output_report)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return 0


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__" and not is_notebook_runtime():
    # Menghentikan proses dan memberi pesan kesalahan yang jelas.
    raise SystemExit(main())
