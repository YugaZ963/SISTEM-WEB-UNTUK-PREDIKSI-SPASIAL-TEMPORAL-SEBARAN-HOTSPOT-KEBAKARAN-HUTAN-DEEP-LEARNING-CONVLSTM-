# File anotasi dari `tools/experiment_historical_risk_patch_improvements.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: kode pendukung project skripsi.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Tool eksperimen tambahan untuk menganalisis class imbalance, patch lebih besar, dan baseline sederhana.

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
import re
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from dataclasses import dataclass
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from datetime import datetime
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_DATASET_DIR = Path("Ipynb") / "Dataset History Fire Hotspot In Riau Province PNG"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_OUTPUT_DIR = Path("artifacts") / "experiments" / "historical_risk_patch_improvements"


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@dataclass(frozen=True)
# Membuat wadah bernama `DatasetRecord` untuk menyimpan data atau aturan kerja.
class DatasetRecord:
    # Menjelaskan data `path` yang disimpan atau dikirim pada bagian ini.
    path: Path
    # Menjelaskan data `date` yang disimpan atau dikirim pada bagian ini.
    date: datetime


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
    return tuple(extensions) if extensions else (".png", ".jpg", ".jpeg")


# Membuat langkah kerja bernama `load_records`.
def load_records(dataset_dir: Path, image_extensions: tuple[str, ...]) -> list[DatasetRecord]:
    # Menyimpan nilai ke `records` untuk dipakai pada langkah berikutnya.
    records: list[DatasetRecord] = []
    # Menyimpan nilai ke `allowed` untuk dipakai pada langkah berikutnya.
    allowed = set(image_extensions)
    # Mengulang proses untuk setiap data dalam daftar.
    for path in sorted(item for item in dataset_dir.iterdir() if item.is_file()):
        # Mengecek syarat sebelum melanjutkan proses.
        if path.suffix.lower() not in allowed:
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
    # Menyimpan nilai ke `valid` untuk dipakai pada langkah berikutnya.
    valid: list[DatasetRecord] = []
    # Menyimpan nilai ke `skipped` untuk dipakai pada langkah berikutnya.
    skipped: list[dict[str, str]] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for record in records:
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
            with Image.open(record.path) as image:
                # Melanjutkan langkah kerja pada bagian kode ini.
                image.verify()
            # Melanjutkan langkah kerja pada bagian kode ini.
            valid.append(record)
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception as exc:
            # Melanjutkan langkah kerja pada bagian kode ini.
            skipped.append({"path": str(record.path), "reason": str(exc)})
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return valid, skipped


# Membuat langkah kerja bernama `build_sample_starts`.
def build_sample_starts(record_count: int, seq_length: int) -> list[int]:
    # Mengecek syarat sebelum melanjutkan proses.
    if record_count <= seq_length:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return []
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return list(range(record_count - seq_length))


# Membuat langkah kerja bernama `split_sample_starts`.
def split_sample_starts(
    # Menjelaskan data `sample_starts` yang disimpan atau dikirim pada bagian ini.
    sample_starts: list[int],
    # Menjelaskan data `train_ratio` yang disimpan atau dikirim pada bagian ini.
    train_ratio: float,
    # Menjelaskan data `val_ratio` yang disimpan atau dikirim pada bagian ini.
    val_ratio: float,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tuple[list[int], list[int], list[int]]:
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
        raise ValueError("Split train/val/test gagal. Periksa jumlah data atau rasio split.")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return train, val, test


# Membuat langkah kerja bernama `normalize_kernel`.
def normalize_kernel(size: int) -> int:
    # Menyimpan nilai ke `size` untuk dipakai pada langkah berikutnya.
    size = max(1, int(size))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return size if size % 2 == 1 else size + 1


# Membuat langkah kerja bernama `extract_red_mask`.
def extract_red_mask(path: Path, dilation_kernel: int) -> np.ndarray:
    # Menyimpan nilai ke `dilation_kernel` untuk dipakai pada langkah berikutnya.
    dilation_kernel = normalize_kernel(dilation_kernel)
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with Image.open(path) as image:
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
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.asarray(mask_image, dtype=np.float32) / 255.0


# Membuat langkah kerja bernama `build_risk_map`.
def build_risk_map(path: Path, dilation_kernel: int, blur_radius: float) -> np.ndarray:
    # Menyimpan nilai ke `risk` untuk dipakai pada langkah berikutnya.
    risk = Image.fromarray(np.uint8(extract_red_mask(path, dilation_kernel=1) * 255.0))
    # Menyimpan nilai ke `dilation_kernel` untuk dipakai pada langkah berikutnya.
    dilation_kernel = normalize_kernel(dilation_kernel)
    # Mengecek syarat sebelum melanjutkan proses.
    if dilation_kernel > 1:
        # Menyimpan nilai ke `risk` untuk dipakai pada langkah berikutnya.
        risk = risk.filter(ImageFilter.MaxFilter(size=dilation_kernel))
    # Mengecek syarat sebelum melanjutkan proses.
    if blur_radius > 0:
        # Menyimpan nilai ke `risk` untuk dipakai pada langkah berikutnya.
        risk = risk.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.asarray(risk, dtype=np.float32) / 255.0


# Membuat langkah kerja bernama `crop_array`.
def crop_array(
    # Menjelaskan data `array` yang disimpan atau dikirim pada bagian ini.
    array: np.ndarray,
    # Menjelaskan data `cy` yang disimpan atau dikirim pada bagian ini.
    cy: int,
    # Menjelaskan data `cx` yang disimpan atau dikirim pada bagian ini.
    cx: int,
    # Menjelaskan data `crop_height` yang disimpan atau dikirim pada bagian ini.
    crop_height: int,
    # Menjelaskan data `crop_width` yang disimpan atau dikirim pada bagian ini.
    crop_width: int,
    # Menyimpan nilai ke `output_height` untuk dipakai pada langkah berikutnya.
    output_height: int | None = None,
    # Menyimpan nilai ke `output_width` untuk dipakai pada langkah berikutnya.
    output_width: int | None = None,
    # Menyimpan nilai ke `resample` untuk dipakai pada langkah berikutnya.
    resample: int = Image.Resampling.BILINEAR,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> np.ndarray:
    # Mengecek syarat sebelum melanjutkan proses.
    if array.ndim == 2:
        # Menyimpan nilai ke `array` untuk dipakai pada langkah berikutnya.
        array = array[..., None]

    # Menyimpan nilai ke `height, width` untuk dipakai pada langkah berikutnya.
    height, width = array.shape[:2]
    # Menyimpan nilai ke `half_h` untuk dipakai pada langkah berikutnya.
    half_h = crop_height // 2
    # Menyimpan nilai ke `half_w` untuk dipakai pada langkah berikutnya.
    half_w = crop_width // 2
    # Menyimpan nilai ke `top` untuk dipakai pada langkah berikutnya.
    top = cy - half_h
    # Menyimpan nilai ke `left` untuk dipakai pada langkah berikutnya.
    left = cx - half_w
    # Menyimpan nilai ke `bottom` untuk dipakai pada langkah berikutnya.
    bottom = top + crop_height
    # Menyimpan nilai ke `right` untuk dipakai pada langkah berikutnya.
    right = left + crop_width

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
        bottom = top + crop_height
        # Menyimpan nilai ke `right` untuk dipakai pada langkah berikutnya.
        right = left + crop_width

    # Menyimpan nilai ke `patch` untuk dipakai pada langkah berikutnya.
    patch = array[top:bottom, left:right, :].astype(np.float32)
    # Mengecek syarat sebelum melanjutkan proses.
    if output_height is None or output_width is None or (output_height == crop_height and output_width == crop_width):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return patch

    # Menyimpan nilai ke `resized_channels` untuk dipakai pada langkah berikutnya.
    resized_channels: list[np.ndarray] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for channel_idx in range(patch.shape[-1]):
        # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
        image = Image.fromarray(np.uint8(np.clip(patch[:, :, channel_idx], 0.0, 1.0) * 255.0))
        # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
        image = image.resize((output_width, output_height), resample)
        # Menyimpan nilai ke `resized_channels.append(np.asarray(image, dtype` untuk dipakai pada langkah berikutnya.
        resized_channels.append(np.asarray(image, dtype=np.float32) / 255.0)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.stack(resized_channels, axis=-1)


# Membuat langkah kerja bernama `sample_patch_centers`.
def sample_patch_centers(
    # Menjelaskan data `target` yang disimpan atau dikirim pada bagian ini.
    target: np.ndarray,
    # Menjelaskan data `positive_count` yang disimpan atau dikirim pada bagian ini.
    positive_count: int,
    # Menjelaskan data `negative_count` yang disimpan atau dikirim pada bagian ini.
    negative_count: int,
    # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
    ground_truth_threshold: float,
    # Menjelaskan data `rng` yang disimpan atau dikirim pada bagian ini.
    rng: np.random.Generator,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> list[tuple[int, int, str]]:
    # Menyimpan nilai ke `binary` untuk dipakai pada langkah berikutnya.
    binary = target >= ground_truth_threshold
    # Menyimpan nilai ke `positive_coords` untuk dipakai pada langkah berikutnya.
    positive_coords = np.argwhere(binary)
    # Menyimpan nilai ke `centers` untuk dipakai pada langkah berikutnya.
    centers: list[tuple[int, int, str]] = []

    # Mengecek syarat sebelum melanjutkan proses.
    if len(positive_coords) > 0 and positive_count > 0:
        # Menyimpan nilai ke `replace` untuk dipakai pada langkah berikutnya.
        replace = len(positive_coords) < positive_count
        # Menyimpan nilai ke `indices` untuk dipakai pada langkah berikutnya.
        indices = rng.choice(len(positive_coords), size=positive_count, replace=replace)
        # Mengulang proses untuk setiap data dalam daftar.
        for idx in np.atleast_1d(indices):
            # Menyimpan nilai ke `cy, cx` untuk dipakai pada langkah berikutnya.
            cy, cx = positive_coords[int(idx)]
            # Melanjutkan langkah kerja pada bagian kode ini.
            centers.append((int(cy), int(cx), "positive"))

    # Menyimpan nilai ke `attempts` untuk dipakai pada langkah berikutnya.
    attempts = max(negative_count * 60, 60)
    # Mengulang proses selama syaratnya masih terpenuhi.
    while negative_count > 0 and attempts > 0:
        # Menyimpan nilai ke `cy` untuk dipakai pada langkah berikutnya.
        cy = int(rng.integers(0, target.shape[0]))
        # Menyimpan nilai ke `cx` untuk dipakai pada langkah berikutnya.
        cx = int(rng.integers(0, target.shape[1]))
        # Mengecek syarat sebelum melanjutkan proses.
        if not binary[cy, cx]:
            # Melanjutkan langkah kerja pada bagian kode ini.
            centers.append((cy, cx, "negative"))
            # Menyimpan nilai ke `negative_count -` untuk dipakai pada langkah berikutnya.
            negative_count -= 1
        # Menyimpan nilai ke `attempts -` untuk dipakai pada langkah berikutnya.
        attempts -= 1

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return centers


# Membuat langkah kerja bernama `metrics_from_counts`.
def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
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
        "tp": int(tp),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "fp": int(fp),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "fn": int(fn),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tn": int(tn),
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


# Membuat langkah kerja bernama `count_binary_metrics`.
def count_binary_metrics(
    # Menjelaskan data `truth_map` yang disimpan atau dikirim pada bagian ini.
    truth_map: np.ndarray,
    # Menjelaskan data `pred_map` yang disimpan atau dikirim pada bagian ini.
    pred_map: np.ndarray,
    # Menjelaskan data `threshold` yang disimpan atau dikirim pada bagian ini.
    threshold: float,
    # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
    ground_truth_threshold: float,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tuple[int, int, int, int]:
    # Menyimpan nilai ke `truth` untuk dipakai pada langkah berikutnya.
    truth = truth_map >= ground_truth_threshold
    # Menyimpan nilai ke `pred` untuk dipakai pada langkah berikutnya.
    pred = pred_map >= threshold
    # Menyimpan nilai ke `tp` untuk dipakai pada langkah berikutnya.
    tp = int(np.logical_and(pred, truth).sum())
    # Menyimpan nilai ke `fp` untuk dipakai pada langkah berikutnya.
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    # Menyimpan nilai ke `fn` untuk dipakai pada langkah berikutnya.
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    # Menyimpan nilai ke `tn` untuk dipakai pada langkah berikutnya.
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tp, fp, fn, tn


# Membuat langkah kerja bernama `dilate_bool`.
def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    # Mengecek syarat sebelum melanjutkan proses.
    if radius <= 0:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return mask.astype(bool)
    # Menyimpan nilai ke `kernel` untuk dipakai pada langkah berikutnya.
    kernel = normalize_kernel((int(radius) * 2) + 1)
    # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
    image = Image.fromarray((mask.astype(bool).astype(np.uint8)) * 255)
    # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
    image = image.filter(ImageFilter.MaxFilter(size=kernel))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.asarray(image, dtype=np.uint8) > 0


# Membuat langkah kerja bernama `count_buffered_metrics`.
def count_buffered_metrics(
    # Menjelaskan data `truth_map` yang disimpan atau dikirim pada bagian ini.
    truth_map: np.ndarray,
    # Menjelaskan data `pred_map` yang disimpan atau dikirim pada bagian ini.
    pred_map: np.ndarray,
    # Menjelaskan data `threshold` yang disimpan atau dikirim pada bagian ini.
    threshold: float,
    # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
    ground_truth_threshold: float,
    # Menjelaskan data `buffer_radius` yang disimpan atau dikirim pada bagian ini.
    buffer_radius: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tuple[int, int, int, int]:
    # Menyimpan nilai ke `truth` untuk dipakai pada langkah berikutnya.
    truth = dilate_bool(truth_map >= ground_truth_threshold, buffer_radius)
    # Menyimpan nilai ke `pred` untuk dipakai pada langkah berikutnya.
    pred = dilate_bool(pred_map >= threshold, buffer_radius)
    # Menyimpan nilai ke `tp` untuk dipakai pada langkah berikutnya.
    tp = int(np.logical_and(pred, truth).sum())
    # Menyimpan nilai ke `fp` untuk dipakai pada langkah berikutnya.
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    # Menyimpan nilai ke `fn` untuk dipakai pada langkah berikutnya.
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    # Menyimpan nilai ke `tn` untuk dipakai pada langkah berikutnya.
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tp, fp, fn, tn


# Membuat langkah kerja bernama `threshold_values`.
def threshold_values(step: float) -> list[float]:
    # Mengecek syarat sebelum melanjutkan proses.
    if step <= 0 or step >= 1:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("threshold-step harus berada pada rentang 0 sampai 1.")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return [round(float(value), 4) for value in np.arange(step, 1.0, step)]


# Membuat langkah kerja bernama `metric_rank`.
def metric_rank(metrics: dict[str, float | int]) -> tuple[float, float, float, float, int]:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return (
        # Melanjutkan langkah kerja pada bagian kode ini.
        float(metrics["f1_score"]),
        # Melanjutkan langkah kerja pada bagian kode ini.
        float(metrics["iou"]),
        # Melanjutkan langkah kerja pada bagian kode ini.
        float(metrics["precision"]),
        # Melanjutkan langkah kerja pada bagian kode ini.
        float(metrics["recall"]),
        # Melanjutkan langkah kerja pada bagian kode ini.
        -int(metrics["fp"]),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )


# Membuat langkah kerja bernama `prepare_records`.
def prepare_records(args: argparse.Namespace) -> tuple[list[DatasetRecord], list[int], list[int], list[int]]:
    # Menyimpan nilai ke `dataset_dir` untuk dipakai pada langkah berikutnya.
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    # Menyimpan nilai ke `records, skipped` untuk dipakai pada langkah berikutnya.
    records, skipped = validate_records(load_records(dataset_dir, parse_image_extensions(args.image_extensions)))
    # Mengecek syarat sebelum melanjutkan proses.
    if skipped:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[warning] {len(skipped)} file dilewati karena tidak valid.")
    # Mengecek syarat sebelum melanjutkan proses.
    if len(records) <= args.seq_length:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Jumlah record tidak cukup untuk membentuk sequence.")
    # Menyimpan nilai ke `sample_starts` untuk dipakai pada langkah berikutnya.
    sample_starts = build_sample_starts(len(records), args.seq_length)
    # Menyimpan nilai ke `train, val, test` untuk dipakai pada langkah berikutnya.
    train, val, test = split_sample_starts(sample_starts, args.train_ratio, args.val_ratio)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return records, train, val, test


# Membuat langkah kerja bernama `select_split`.
def select_split(args: argparse.Namespace, train: list[int], val: list[int], test: list[int]) -> list[int]:
    # Menyimpan nilai ke `split_map` untuk dipakai pada langkah berikutnya.
    split_map = {"train": train, "val": val, "test": test}
    # Menyimpan nilai ke `selected` untuk dipakai pada langkah berikutnya.
    selected = split_map[args.split]
    # Mengecek syarat sebelum melanjutkan proses.
    if args.max_samples and args.max_samples > 0:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return selected[: args.max_samples]
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return selected


# Membuat langkah kerja bernama `run_imbalance`.
def run_imbalance(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `records, train, val, test` untuk dipakai pada langkah berikutnya.
    records, train, val, test = prepare_records(args)
    # Menyimpan nilai ke `selected` untuk dipakai pada langkah berikutnya.
    selected = select_split(args, train, val, test)
    # Menyimpan nilai ke `rng` untuk dipakai pada langkah berikutnya.
    rng = np.random.default_rng(args.seed)
    # Menyimpan nilai ke `patch_sizes` untuk dipakai pada langkah berikutnya.
    patch_sizes = [int(item.strip()) for item in args.patch_sizes.split(",") if item.strip()]

    # Menyimpan nilai ke `positive_pixels` untuk dipakai pada langkah berikutnya.
    positive_pixels = 0
    # Menyimpan nilai ke `total_pixels` untuk dipakai pada langkah berikutnya.
    total_pixels = 0
    # Menyimpan nilai ke `patch_stats` untuk dipakai pada langkah berikutnya.
    patch_stats = {
        # Menjelaskan data `size` yang disimpan atau dikirim pada bagian ini.
        size: {"positive_patch_ratios": [], "negative_patch_ratios": []}
        # Mengulang proses untuk setiap data dalam daftar.
        for size in patch_sizes
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }

    # Mengulang proses untuk setiap data dalam daftar.
    for start in selected:
        # Menyimpan nilai ke `target_record` untuk dipakai pada langkah berikutnya.
        target_record = records[start + args.seq_length]
        # Menyimpan nilai ke `target` untuk dipakai pada langkah berikutnya.
        target = build_risk_map(target_record.path, args.label_dilation_kernel, args.label_blur_radius)
        # Menyimpan nilai ke `binary` untuk dipakai pada langkah berikutnya.
        binary = target >= args.ground_truth_threshold
        # Menyimpan nilai ke `positive_pixels +` untuk dipakai pada langkah berikutnya.
        positive_pixels += int(binary.sum())
        # Menyimpan nilai ke `total_pixels +` untuk dipakai pada langkah berikutnya.
        total_pixels += int(binary.size)

        # Menyimpan nilai ke `centers` untuk dipakai pada langkah berikutnya.
        centers = sample_patch_centers(
            # Menyimpan nilai ke `target` untuk dipakai pada langkah berikutnya.
            target=target,
            # Menyimpan nilai ke `positive_count` untuk dipakai pada langkah berikutnya.
            positive_count=args.train_positive_patches,
            # Menyimpan nilai ke `negative_count` untuk dipakai pada langkah berikutnya.
            negative_count=args.train_negative_patches,
            # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
            ground_truth_threshold=args.ground_truth_threshold,
            # Menyimpan nilai ke `rng` untuk dipakai pada langkah berikutnya.
            rng=rng,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Mengulang proses untuk setiap data dalam daftar.
        for size in patch_sizes:
            # Mengulang proses untuk setiap data dalam daftar.
            for cy, cx, kind in centers:
                # Menyimpan nilai ke `patch` untuk dipakai pada langkah berikutnya.
                patch = crop_array(target, cy, cx, size, size)[:, :, 0]
                # Menyimpan nilai ke `ratio` untuk dipakai pada langkah berikutnya.
                ratio = float((patch >= args.ground_truth_threshold).mean())
                # Mengecek syarat sebelum melanjutkan proses.
                if kind == "positive":
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    patch_stats[size]["positive_patch_ratios"].append(ratio)
                # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
                else:
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    patch_stats[size]["negative_patch_ratios"].append(ratio)

    # Menyimpan nilai ke `negative_pixels` untuk dipakai pada langkah berikutnya.
    negative_pixels = total_pixels - positive_pixels
    # Menyimpan nilai ke `positive_ratio` untuk dipakai pada langkah berikutnya.
    positive_ratio = 0.0 if total_pixels == 0 else positive_pixels / total_pixels
    # Menyimpan nilai ke `raw_pos_weight` untuk dipakai pada langkah berikutnya.
    raw_pos_weight = 0.0 if positive_pixels == 0 else negative_pixels / positive_pixels
    # Menyimpan nilai ke `capped_pos_weight` untuk dipakai pada langkah berikutnya.
    capped_pos_weight = min(raw_pos_weight, args.max_pos_weight)

    # Menyimpan nilai ke `patch_summary` untuk dipakai pada langkah berikutnya.
    patch_summary = {}
    # Mengulang proses untuk setiap data dalam daftar.
    for size, values in patch_stats.items():
        # Menyimpan nilai ke `pos_values` untuk dipakai pada langkah berikutnya.
        pos_values = values["positive_patch_ratios"]
        # Menyimpan nilai ke `neg_values` untuk dipakai pada langkah berikutnya.
        neg_values = values["negative_patch_ratios"]
        # Menyimpan nilai ke `patch_summary[str(size)]` untuk dipakai pada langkah berikutnya.
        patch_summary[str(size)] = {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "positive_patch_count": len(pos_values),
            # Melanjutkan langkah kerja pada bagian kode ini.
            "negative_patch_count": len(neg_values),
            # Melanjutkan langkah kerja pada bagian kode ini.
            "mean_positive_patch_positive_ratio": float(np.mean(pos_values)) if pos_values else 0.0,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "median_positive_patch_positive_ratio": float(np.median(pos_values)) if pos_values else 0.0,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "mean_negative_patch_positive_ratio": float(np.mean(neg_values)) if neg_values else 0.0,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        }

    # Menyimpan nilai ke `report` untuk dipakai pada langkah berikutnya.
    report = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "mode": "imbalance",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "split": args.split,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sample_count": len(selected),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dataset_frame_count": len(records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_samples": len(train),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_samples": len(val),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_samples": len(test),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_pixels": positive_pixels,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "negative_pixels": negative_pixels,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "total_pixels": total_pixels,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_ratio": positive_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "raw_positive_class_weight": raw_pos_weight,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "capped_positive_class_weight": capped_pos_weight,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recommended_framing": "prediksi area risiko hotspot, bukan prediksi titik hotspot presisi tinggi",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_summary": patch_summary,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "practical_recommendation": {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "increase_positive_patch_sampling": "coba 6:1 atau 8:1 untuk train-positive-patches:train-negative-patches",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "larger_patch_candidates": patch_sizes,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "use_feature_stack": "mask_context3 untuk konteks tambahan yang sudah tersedia di script training utama",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Melanjutkan langkah kerja pada bagian kode ini.
    write_report(report, args.output_dir, "imbalance_patch_report.json")
    # Melanjutkan langkah kerja pada bagian kode ini.
    print_json_or_pretty(report, args.json)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return report


# Membuat langkah kerja bernama `run_baseline`.
def run_baseline(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `records, train, val, test` untuk dipakai pada langkah berikutnya.
    records, train, val, test = prepare_records(args)
    # Menyimpan nilai ke `selected` untuk dipakai pada langkah berikutnya.
    selected = select_split(args, train, val, test)
    # Menyimpan nilai ke `thresholds` untuk dipakai pada langkah berikutnya.
    thresholds = threshold_values(args.threshold_step)
    # Menyimpan nilai ke `counts` untuk dipakai pada langkah berikutnya.
    counts = {threshold: [0, 0, 0, 0] for threshold in thresholds}

    # Mengulang proses untuk setiap data dalam daftar.
    for start in selected:
        # Menyimpan nilai ke `h0_record` untuk dipakai pada langkah berikutnya.
        h0_record = records[start + args.seq_length - 1]
        # Menyimpan nilai ke `target_record` untuk dipakai pada langkah berikutnya.
        target_record = records[start + args.seq_length]
        # Menyimpan nilai ke `pred` untuk dipakai pada langkah berikutnya.
        pred = build_risk_map(h0_record.path, args.baseline_dilation_kernel, args.baseline_blur_radius)
        # Menyimpan nilai ke `truth` untuk dipakai pada langkah berikutnya.
        truth = build_risk_map(target_record.path, args.label_dilation_kernel, args.label_blur_radius)
        # Mengulang proses untuk setiap data dalam daftar.
        for threshold in thresholds:
            # Menyimpan nilai ke `tp, fp, fn, tn` untuk dipakai pada langkah berikutnya.
            tp, fp, fn, tn = count_binary_metrics(truth, pred, threshold, args.ground_truth_threshold)
            # Menyimpan nilai ke `bucket` untuk dipakai pada langkah berikutnya.
            bucket = counts[threshold]
            # Menyimpan nilai ke `bucket[0] +` untuk dipakai pada langkah berikutnya.
            bucket[0] += tp
            # Menyimpan nilai ke `bucket[1] +` untuk dipakai pada langkah berikutnya.
            bucket[1] += fp
            # Menyimpan nilai ke `bucket[2] +` untuk dipakai pada langkah berikutnya.
            bucket[2] += fn
            # Menyimpan nilai ke `bucket[3] +` untuk dipakai pada langkah berikutnya.
            bucket[3] += tn

    # Menyimpan nilai ke `sweep` untuk dipakai pada langkah berikutnya.
    sweep = []
    # Menyimpan nilai ke `best_threshold` untuk dipakai pada langkah berikutnya.
    best_threshold = thresholds[0]
    # Menyimpan nilai ke `best_metrics` untuk dipakai pada langkah berikutnya.
    best_metrics: dict[str, float | int] | None = None
    # Mengulang proses untuk setiap data dalam daftar.
    for threshold in thresholds:
        # Menyimpan nilai ke `tp, fp, fn, tn` untuk dipakai pada langkah berikutnya.
        tp, fp, fn, tn = counts[threshold]
        # Menyimpan nilai ke `metrics` untuk dipakai pada langkah berikutnya.
        metrics = metrics_from_counts(tp, fp, fn, tn)
        # Menyimpan nilai ke `row` untuk dipakai pada langkah berikutnya.
        row = {"threshold": threshold, **metrics}
        # Melanjutkan langkah kerja pada bagian kode ini.
        sweep.append(row)
        # Mengecek syarat sebelum melanjutkan proses.
        if best_metrics is None or metric_rank(metrics) > metric_rank(best_metrics):
            # Menyimpan nilai ke `best_threshold` untuk dipakai pada langkah berikutnya.
            best_threshold = threshold
            # Menyimpan nilai ke `best_metrics` untuk dipakai pada langkah berikutnya.
            best_metrics = metrics

    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert best_metrics is not None

    # Menyimpan nilai ke `buffered_counts` untuk dipakai pada langkah berikutnya.
    buffered_counts = [0, 0, 0, 0]
    # Mengulang proses untuk setiap data dalam daftar.
    for start in selected:
        # Menyimpan nilai ke `h0_record` untuk dipakai pada langkah berikutnya.
        h0_record = records[start + args.seq_length - 1]
        # Menyimpan nilai ke `target_record` untuk dipakai pada langkah berikutnya.
        target_record = records[start + args.seq_length]
        # Menyimpan nilai ke `pred` untuk dipakai pada langkah berikutnya.
        pred = build_risk_map(h0_record.path, args.baseline_dilation_kernel, args.baseline_blur_radius)
        # Menyimpan nilai ke `truth` untuk dipakai pada langkah berikutnya.
        truth = build_risk_map(target_record.path, args.label_dilation_kernel, args.label_blur_radius)
        # Menyimpan nilai ke `tp, fp, fn, tn` untuk dipakai pada langkah berikutnya.
        tp, fp, fn, tn = count_buffered_metrics(
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            truth,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            pred,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            best_threshold,
            # Melanjutkan langkah kerja pada bagian kode ini.
            args.ground_truth_threshold,
            # Melanjutkan langkah kerja pada bagian kode ini.
            args.buffer_radius,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `buffered_counts[0] +` untuk dipakai pada langkah berikutnya.
        buffered_counts[0] += tp
        # Menyimpan nilai ke `buffered_counts[1] +` untuk dipakai pada langkah berikutnya.
        buffered_counts[1] += fp
        # Menyimpan nilai ke `buffered_counts[2] +` untuk dipakai pada langkah berikutnya.
        buffered_counts[2] += fn
        # Menyimpan nilai ke `buffered_counts[3] +` untuk dipakai pada langkah berikutnya.
        buffered_counts[3] += tn

    # Menyimpan nilai ke `buffered_metrics` untuk dipakai pada langkah berikutnya.
    buffered_metrics = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "buffer_radius": args.buffer_radius,
        # Melanjutkan langkah kerja pada bagian kode ini.
        **metrics_from_counts(*buffered_counts),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Menyimpan nilai ke `report` untuk dipakai pada langkah berikutnya.
    report = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "mode": "persistence_baseline",
        # Menyimpan nilai ke `"baseline_definition"` untuk dipakai pada langkah berikutnya.
        "baseline_definition": "prediksi H+1 = risk map dari H0 yang diberi dilation/blur",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "split": args.split,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sample_count": len(selected),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "best_threshold": best_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "best_standard_metrics": best_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "best_buffered_metrics": buffered_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sweep": sweep,
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "interpretation": (
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Jika ConvLSTM tidak jauh lebih baik dari baseline ini, maka model belum menangkap "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "pola temporal yang kuat di luar persistensi hotspot H0."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Melanjutkan langkah kerja pada bagian kode ini.
    write_report(report, args.output_dir, "h0_persistence_baseline_report.json")
    # Melanjutkan langkah kerja pada bagian kode ini.
    print_json_or_pretty(report, args.json)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return report


# Membuat langkah kerja bernama `find_record_with_hotspot`.
def find_record_with_hotspot(records: list[DatasetRecord], dilation_kernel: int) -> DatasetRecord:
    # Menyimpan nilai ke `best_record` untuk dipakai pada langkah berikutnya.
    best_record: DatasetRecord | None = None
    # Menyimpan nilai ke `best_count` untuk dipakai pada langkah berikutnya.
    best_count = -1
    # Mengulang proses untuk setiap data dalam daftar.
    for record in records:
        # Menggabungkan hasil deteksi merah menjadi mask hotspot.
        mask = extract_red_mask(record.path, dilation_kernel)
        # Menyimpan nilai ke `count` untuk dipakai pada langkah berikutnya.
        count = int((mask > 0).sum())
        # Mengecek syarat sebelum melanjutkan proses.
        if count > best_count:
            # Menyimpan nilai ke `best_record` untuk dipakai pada langkah berikutnya.
            best_record = record
            # Menyimpan nilai ke `best_count` untuk dipakai pada langkah berikutnya.
            best_count = count
    # Mengecek syarat sebelum melanjutkan proses.
    if best_record is None or best_count <= 0:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Tidak ditemukan citra dengan hotspot merah.")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return best_record


# Membuat langkah kerja bernama `draw_centered_text`.
def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: tuple[int, int, int]) -> None:
    # Menyimpan nilai ke `font` untuk dipakai pada langkah berikutnya.
    font = ImageFont.load_default()
    # Menyimpan nilai ke `bbox` untuk dipakai pada langkah berikutnya.
    bbox = draw.textbbox((0, 0), text, font=font)
    # Menyimpan nilai ke `text_width` untuk dipakai pada langkah berikutnya.
    text_width = bbox[2] - bbox[0]
    # Menyimpan nilai ke `text_height` untuk dipakai pada langkah berikutnya.
    text_height = bbox[3] - bbox[1]
    # Menyimpan nilai ke `x0, y0, x1, y1` untuk dipakai pada langkah berikutnya.
    x0, y0, x1, y1 = box
    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    draw.text(
        # Melanjutkan langkah kerja pada bagian kode ini.
        (x0 + (x1 - x0 - text_width) / 2, y0 + (y1 - y0 - text_height) / 2),
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        text,
        # Menyimpan nilai ke `fill` untuk dipakai pada langkah berikutnya.
        fill=fill,
        # Menyimpan nilai ke `font` untuk dipakai pada langkah berikutnya.
        font=font,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )


# Membuat langkah kerja bernama `save_multiscale_preview`.
def save_multiscale_preview(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `records, train, val, test` untuk dipakai pada langkah berikutnya.
    records, train, val, test = prepare_records(args)
    # Menyimpan nilai ke `source` untuk dipakai pada langkah berikutnya.
    source = Path(args.source_image).expanduser().resolve() if args.source_image else None
    # Mengecek syarat sebelum melanjutkan proses.
    if source and source.exists():
        # Menyimpan nilai ke `record` untuk dipakai pada langkah berikutnya.
        record = DatasetRecord(path=source, date=datetime.min)
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menyimpan nilai ke `record` untuk dipakai pada langkah berikutnya.
        record = find_record_with_hotspot(records, args.input_dilation_kernel)

    # Menggabungkan hasil deteksi merah menjadi mask hotspot.
    mask = extract_red_mask(record.path, args.input_dilation_kernel)
    # Menyimpan nilai ke `positive_coords` untuk dipakai pada langkah berikutnya.
    positive_coords = np.argwhere(mask >= args.ground_truth_threshold)
    # Mengecek syarat sebelum melanjutkan proses.
    if len(positive_coords) == 0:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Citra terpilih tidak punya piksel hotspot setelah threshold.")
    # Menyimpan nilai ke `center_idx` untuk dipakai pada langkah berikutnya.
    center_idx = int(len(positive_coords) // 2)
    # Menyimpan nilai ke `cy, cx` untuk dipakai pada langkah berikutnya.
    cy, cx = positive_coords[center_idx]
    # Menyimpan nilai ke `cy` untuk dipakai pada langkah berikutnya.
    cy = int(cy)
    # Menyimpan nilai ke `cx` untuk dipakai pada langkah berikutnya.
    cx = int(cx)

    # Menyimpan nilai ke `patch_size` untuk dipakai pada langkah berikutnya.
    patch_size = int(args.patch_size)
    # Menyimpan nilai ke `context_size` untuk dipakai pada langkah berikutnya.
    context_size = int(round(patch_size * float(args.context_scale)))
    # Menyimpan nilai ke `local_mask` untuk dipakai pada langkah berikutnya.
    local_mask = crop_array(mask, cy, cx, patch_size, patch_size, patch_size, patch_size, Image.Resampling.NEAREST)
    # Menyimpan nilai ke `context_mask` untuk dipakai pada langkah berikutnya.
    context_mask = crop_array(mask, cy, cx, context_size, context_size, patch_size, patch_size, Image.Resampling.BILINEAR)
    # Menyimpan nilai ke `multiscale` untuk dipakai pada langkah berikutnya.
    multiscale = np.concatenate([local_mask, context_mask], axis=-1)

    # Menyimpan nilai ke `output_dir` untuk dipakai pada langkah berikutnya.
    output_dir = Path(args.output_dir).expanduser().resolve()
    # Menyimpan nilai ke `output_dir.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Menyimpan nilai ke `local_path` untuk dipakai pada langkah berikutnya.
    local_path = output_dir / f"multiscale_local_mask_{patch_size}.png"
    # Menyimpan nilai ke `context_path` untuk dipakai pada langkah berikutnya.
    context_path = output_dir / f"multiscale_context_mask_{patch_size}_scale_{args.context_scale}.png"
    # Menyimpan nilai ke `stacked_path` untuk dipakai pada langkah berikutnya.
    stacked_path = output_dir / f"multiscale_two_channel_preview_{patch_size}.png"
    # Menyimpan nilai ke `composite_path` untuk dipakai pada langkah berikutnya.
    composite_path = output_dir / f"multiscale_patch_preview_{patch_size}.png"

    # Mengolah file gambar, seperti membuka atau mengubah citra hotspot.
    Image.fromarray(np.uint8(local_mask[:, :, 0] * 255)).save(local_path)
    # Mengolah file gambar, seperti membuka atau mengubah citra hotspot.
    Image.fromarray(np.uint8(context_mask[:, :, 0] * 255)).save(context_path)

    # Menyimpan nilai ke `stacked_rgb` untuk dipakai pada langkah berikutnya.
    stacked_rgb = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    # Menyimpan nilai ke `stacked_rgb[` untuk dipakai pada langkah berikutnya.
    stacked_rgb[:, :, 0] = np.uint8(local_mask[:, :, 0] * 255)
    # Menyimpan nilai ke `stacked_rgb[` untuk dipakai pada langkah berikutnya.
    stacked_rgb[:, :, 1] = np.uint8(context_mask[:, :, 0] * 255)
    # Mengolah file gambar, seperti membuka atau mengubah citra hotspot.
    Image.fromarray(stacked_rgb).save(stacked_path)

    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with Image.open(record.path) as image:
        # Menyimpan nilai ke `source_rgb` untuk dipakai pada langkah berikutnya.
        source_rgb = image.convert("RGB")
    # Menyimpan nilai ke `preview` untuk dipakai pada langkah berikutnya.
    preview = source_rgb.copy()
    # Menyimpan nilai ke `draw` untuk dipakai pada langkah berikutnya.
    draw = ImageDraw.Draw(preview, "RGBA")

    # Membuat langkah kerja bernama `box`.
    def box(size: int) -> tuple[int, int, int, int]:
        # Menyimpan nilai ke `left` untuk dipakai pada langkah berikutnya.
        left = max(0, cx - size // 2)
        # Menyimpan nilai ke `top` untuk dipakai pada langkah berikutnya.
        top = max(0, cy - size // 2)
        # Menyimpan nilai ke `right` untuk dipakai pada langkah berikutnya.
        right = min(preview.width, left + size)
        # Menyimpan nilai ke `bottom` untuk dipakai pada langkah berikutnya.
        bottom = min(preview.height, top + size)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return left, top, right, bottom

    # Menyimpan nilai ke `draw.rectangle(box(context_size), outline` untuk dipakai pada langkah berikutnya.
    draw.rectangle(box(context_size), outline=(255, 220, 0, 255), width=6)
    # Menyimpan nilai ke `draw.rectangle(box(patch_size), outline` untuk dipakai pada langkah berikutnya.
    draw.rectangle(box(patch_size), outline=(255, 0, 0, 255), width=6)
    # Melanjutkan langkah kerja pada bagian kode ini.
    preview.thumbnail((920, 460), Image.Resampling.LANCZOS)

    # Menyimpan nilai ke `panel_w` untuk dipakai pada langkah berikutnya.
    panel_w = patch_size
    # Menyimpan nilai ke `header_h` untuk dipakai pada langkah berikutnya.
    header_h = 34
    # Menyimpan nilai ke `composite` untuk dipakai pada langkah berikutnya.
    composite = Image.new("RGB", (max(920, panel_w * 3), preview.height + panel_w + header_h * 2 + 36), (242, 246, 250))
    # Menyimpan nilai ke `d` untuk dipakai pada langkah berikutnya.
    d = ImageDraw.Draw(composite)
    # Melanjutkan langkah kerja pada bagian kode ini.
    composite.paste(preview, (0, 0))
    # Menyimpan nilai ke `y` untuk dipakai pada langkah berikutnya.
    y = preview.height + 18
    # Menyimpan nilai ke `labels` untuk dipakai pada langkah berikutnya.
    labels = ["Local detail mask", "Large context mask", "2-channel preview"]
    # Menyimpan nilai ke `images` untuk dipakai pada langkah berikutnya.
    images = [Image.open(local_path).convert("RGB"), Image.open(context_path).convert("RGB"), Image.open(stacked_path).convert("RGB")]
    # Mengulang proses untuk setiap data dalam daftar.
    for idx, (label, item) in enumerate(zip(labels, images)):
        # Menyimpan nilai ke `x` untuk dipakai pada langkah berikutnya.
        x = idx * panel_w
        # Menyimpan nilai ke `d.rectangle((x, y, x + panel_w, y + header_h), fill` untuk dipakai pada langkah berikutnya.
        d.rectangle((x, y, x + panel_w, y + header_h), fill=(30, 68, 102))
        # Melanjutkan langkah kerja pada bagian kode ini.
        draw_centered_text(d, (x, y, x + panel_w, y + header_h), label, (255, 255, 255))
        # Melanjutkan langkah kerja pada bagian kode ini.
        composite.paste(item, (x, y + header_h))
    # Melanjutkan langkah kerja pada bagian kode ini.
    composite.save(composite_path)

    # Menyimpan nilai ke `report` untuk dipakai pada langkah berikutnya.
    report = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "mode": "preview_multiscale",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "source_image": str(record.path),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_size": patch_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "context_size": context_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "context_scale": args.context_scale,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "multiscale_tensor_shape": list(multiscale.shape),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "meaning": {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "channel_0": "mask hotspot lokal untuk detail",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "channel_1": "mask hotspot konteks lebih luas yang di-resize ke ukuran patch",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Melanjutkan langkah kerja pada bagian kode ini.
        "outputs": {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "local_mask": str(local_path),
            # Melanjutkan langkah kerja pada bagian kode ini.
            "context_mask": str(context_path),
            # Melanjutkan langkah kerja pada bagian kode ini.
            "two_channel_preview": str(stacked_path),
            # Melanjutkan langkah kerja pada bagian kode ini.
            "composite": str(composite_path),
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Melanjutkan langkah kerja pada bagian kode ini.
    write_report(report, args.output_dir, "multiscale_preview_report.json")
    # Melanjutkan langkah kerja pada bagian kode ini.
    print_json_or_pretty(report, args.json)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return report


# Membuat langkah kerja bernama `run_commands`.
def run_commands(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `dataset_dir` untuk dipakai pada langkah berikutnya.
    dataset_dir = args.dataset_dir
    # Menyimpan nilai ke `commands` untuk dipakai pada langkah berikutnya.
    commands = {
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "baseline": (
            # Melanjutkan langkah kerja pada bagian kode ini.
            'python -u tools/experiment_historical_risk_patch_improvements.py baseline '
            # Melanjutkan langkah kerja pada bagian kode ini.
            f'--dataset-dir "{dataset_dir}" --split test --threshold-step 0.05'
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "imbalance_patch_224_preview": (
            # Melanjutkan langkah kerja pada bagian kode ini.
            'python -u tools/experiment_historical_risk_patch_improvements.py imbalance '
            # Melanjutkan langkah kerja pada bagian kode ini.
            f'--dataset-dir "{dataset_dir}" --patch-sizes 160,224,256 '
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--train-positive-patches 6 --train-negative-patches 1 --split train --max-samples 120"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "multiscale_preview": (
            # Melanjutkan langkah kerja pada bagian kode ini.
            'python -u tools/experiment_historical_risk_patch_improvements.py preview-multiscale '
            # Melanjutkan langkah kerja pada bagian kode ini.
            f'--dataset-dir "{dataset_dir}" --patch-size 224 --context-scale 2.0'
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "training_patch_224_wsl": (
            # Melanjutkan langkah kerja pada bagian kode ini.
            "bash run_train_historical_risk_patch_wsl_gpu.sh "
            # Melanjutkan langkah kerja pada bagian kode ini.
            f'--dataset-dir "$HOME/projects/sistem-web-skripsi-ta/Ipynb/Dataset History Fire Hotspot In Riau Province PNG" '
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--image-extensions .png --patch-width 224 --patch-height 224 --batch-size 1 --epochs 3 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--train-positive-patches 6 --train-negative-patches 1 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--input-dilation-kernel 5 --label-dilation-kernel 9 --label-blur-radius 2.0 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--evaluation-buffer-radius 5 --threshold 0.55 --ground-truth-threshold 0.05 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--loss-strategy wbce_dice_context --feature-stack mask_context3 --context-kernel 15 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--enable-augmentation --augmentation-rotate90"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "training_patch_256_wsl": (
            # Melanjutkan langkah kerja pada bagian kode ini.
            "bash run_train_historical_risk_patch_wsl_gpu.sh "
            # Melanjutkan langkah kerja pada bagian kode ini.
            f'--dataset-dir "$HOME/projects/sistem-web-skripsi-ta/Ipynb/Dataset History Fire Hotspot In Riau Province PNG" '
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--image-extensions .png --patch-width 256 --patch-height 256 --batch-size 1 --epochs 3 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--train-positive-patches 6 --train-negative-patches 1 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--input-dilation-kernel 5 --label-dilation-kernel 9 --label-blur-radius 2.0 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--evaluation-buffer-radius 5 --threshold 0.55 --ground-truth-threshold 0.05 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--loss-strategy wbce_dice_context --feature-stack mask_context3 --context-kernel 15 "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "--enable-augmentation --augmentation-rotate90"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Melanjutkan langkah kerja pada bagian kode ini.
    print_json_or_pretty(commands, args.json)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return commands


# Membuat langkah kerja bernama `write_report`.
def write_report(report: dict, output_dir: str | Path, file_name: str) -> None:
    # Menyimpan nilai ke `output_path` untuk dipakai pada langkah berikutnya.
    output_path = Path(output_dir).expanduser().resolve()
    # Menyimpan nilai ke `output_path.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_path.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `(output_path / file_name).write_text(json.dumps(report, indent` untuk dipakai pada langkah berikutnya.
    (output_path / file_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# Membuat langkah kerja bernama `print_json_or_pretty`.
def print_json_or_pretty(data: dict, as_json: bool) -> None:
    # Mengecek syarat sebelum melanjutkan proses.
    if as_json:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(json.dumps(data, indent=2, ensure_ascii=False))
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    # Menyimpan nilai ke `mode` untuk dipakai pada langkah berikutnya.
    mode = data.get("mode", "commands")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[experiment] Mode: {mode}")
    # Mengecek syarat sebelum melanjutkan proses.
    if mode == "imbalance":
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[experiment] Split: {data['split']} | samples: {data['sample_count']}")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[experiment] Positive ratio: {data['positive_ratio']:.8f}")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[experiment] Raw positive class weight: {data['raw_positive_class_weight']:.4f}")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[experiment] Capped positive class weight: {data['capped_positive_class_weight']:.4f}")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[experiment] Patch summary:")
        # Mengulang proses untuk setiap data dalam daftar.
        for size, row in data["patch_summary"].items():
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print(
                # Melanjutkan langkah kerja pada bagian kode ini.
                "  - "
                # Menyimpan nilai ke `f"{size}x{size}` untuk dipakai pada langkah berikutnya.
                f"{size}x{size}: pos_patch_mean={row['mean_positive_patch_positive_ratio']:.6f}, "
                # Menyimpan nilai ke `f"neg_patch_mean` untuk dipakai pada langkah berikutnya.
                f"neg_patch_mean={row['mean_negative_patch_positive_ratio']:.6f}"
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return
    # Mengecek syarat sebelum melanjutkan proses.
    if mode == "persistence_baseline":
        # Menyimpan nilai ke `standard` untuk dipakai pada langkah berikutnya.
        standard = data["best_standard_metrics"]
        # Menyimpan nilai ke `buffered` untuk dipakai pada langkah berikutnya.
        buffered = data["best_buffered_metrics"]
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[experiment] Split: {data['split']} | samples: {data['sample_count']}")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[experiment] Best threshold: {data['best_threshold']}")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "[experiment] Standard: "
            # Menyimpan nilai ke `f"precision` untuk dipakai pada langkah berikutnya.
            f"precision={standard['precision']:.4f}, recall={standard['recall']:.4f}, "
            # Menyimpan nilai ke `f"f1` untuk dipakai pada langkah berikutnya.
            f"f1={standard['f1_score']:.4f}, iou={standard['iou']:.4f}"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "[experiment] Buffered: "
            # Menyimpan nilai ke `f"precision` untuk dipakai pada langkah berikutnya.
            f"precision={buffered['precision']:.4f}, recall={buffered['recall']:.4f}, "
            # Menyimpan nilai ke `f"f1` untuk dipakai pada langkah berikutnya.
            f"f1={buffered['f1_score']:.4f}, iou={buffered['iou']:.4f}"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return
    # Mengecek syarat sebelum melanjutkan proses.
    if mode == "preview_multiscale":
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[experiment] Source: {data['source_image']}")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[experiment] Tensor shape: {data['multiscale_tensor_shape']}")
        # Mengulang proses untuk setiap data dalam daftar.
        for key, path in data["outputs"].items():
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print(f"[experiment] {key}: {path}")
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    # Mengulang proses untuk setiap data dalam daftar.
    for key, value in data.items():
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[{key}]")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(value)


# Membuat langkah kerja bernama `add_common_args`.
def add_common_args(parser: argparse.ArgumentParser) -> None:
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--image-extensions", default=".png")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--seq-length", type=int, default=7)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--train-ratio", type=float, default=0.70)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--max-samples", type=int, default=0)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--ground-truth-threshold", type=float, default=0.05)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--label-dilation-kernel", type=int, default=9)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--label-blur-radius", type=float, default=2.0)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--input-dilation-kernel", type=int, default=5)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--json", action="store_true")


# Membuat langkah kerja bernama `build_parser`.
def build_parser() -> argparse.ArgumentParser:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description=(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Eksperimen terpisah untuk peningkatan historical_risk_patch: "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "class imbalance, patch besar, multi-scale preview, baseline H0, dan evaluasi area risiko."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `subparsers` untuk dipakai pada langkah berikutnya.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Menyimpan nilai ke `imbalance` untuk dipakai pada langkah berikutnya.
    imbalance = subparsers.add_parser("imbalance", help="Ringkas class imbalance dan efek kandidat patch besar.")
    # Melanjutkan langkah kerja pada bagian kode ini.
    add_common_args(imbalance)
    # Menyimpan nilai ke `imbalance.add_argument("--patch-sizes", default` untuk dipakai pada langkah berikutnya.
    imbalance.add_argument("--patch-sizes", default="160,224,256")
    # Menyimpan nilai ke `imbalance.add_argument("--train-positive-patches", type` untuk dipakai pada langkah berikutnya.
    imbalance.add_argument("--train-positive-patches", type=int, default=6)
    # Menyimpan nilai ke `imbalance.add_argument("--train-negative-patches", type` untuk dipakai pada langkah berikutnya.
    imbalance.add_argument("--train-negative-patches", type=int, default=1)
    # Menyimpan nilai ke `imbalance.add_argument("--max-pos-weight", type` untuk dipakai pada langkah berikutnya.
    imbalance.add_argument("--max-pos-weight", type=float, default=50.0)
    # Menyimpan nilai ke `imbalance.add_argument("--seed", type` untuk dipakai pada langkah berikutnya.
    imbalance.add_argument("--seed", type=int, default=42)
    # Menyimpan nilai ke `imbalance.set_defaults(func` untuk dipakai pada langkah berikutnya.
    imbalance.set_defaults(func=run_imbalance, split="train")

    # Menyimpan nilai ke `baseline` untuk dipakai pada langkah berikutnya.
    baseline = subparsers.add_parser("baseline", help="Evaluasi baseline sederhana: prediksi H+1 = H0 dilated.")
    # Melanjutkan langkah kerja pada bagian kode ini.
    add_common_args(baseline)
    # Menyimpan nilai ke `baseline.add_argument("--baseline-dilation-kernel", type` untuk dipakai pada langkah berikutnya.
    baseline.add_argument("--baseline-dilation-kernel", type=int, default=9)
    # Menyimpan nilai ke `baseline.add_argument("--baseline-blur-radius", type` untuk dipakai pada langkah berikutnya.
    baseline.add_argument("--baseline-blur-radius", type=float, default=2.0)
    # Menyimpan nilai ke `baseline.add_argument("--threshold-step", type` untuk dipakai pada langkah berikutnya.
    baseline.add_argument("--threshold-step", type=float, default=0.05)
    # Menyimpan nilai ke `baseline.add_argument("--buffer-radius", type` untuk dipakai pada langkah berikutnya.
    baseline.add_argument("--buffer-radius", type=int, default=5)
    # Menyimpan nilai ke `baseline.set_defaults(func` untuk dipakai pada langkah berikutnya.
    baseline.set_defaults(func=run_baseline)

    # Menyimpan nilai ke `preview` untuk dipakai pada langkah berikutnya.
    preview = subparsers.add_parser("preview-multiscale", help="Buat visualisasi patch lokal + konteks lebih luas.")
    # Melanjutkan langkah kerja pada bagian kode ini.
    add_common_args(preview)
    # Menyimpan nilai ke `preview.add_argument("--patch-size", type` untuk dipakai pada langkah berikutnya.
    preview.add_argument("--patch-size", type=int, default=224)
    # Menyimpan nilai ke `preview.add_argument("--context-scale", type` untuk dipakai pada langkah berikutnya.
    preview.add_argument("--context-scale", type=float, default=2.0)
    # Menyimpan nilai ke `preview.add_argument("--source-image", default` untuk dipakai pada langkah berikutnya.
    preview.add_argument("--source-image", default="")
    # Menyimpan nilai ke `preview.set_defaults(func` untuk dipakai pada langkah berikutnya.
    preview.set_defaults(func=save_multiscale_preview)

    # Menyimpan nilai ke `commands` untuk dipakai pada langkah berikutnya.
    commands = subparsers.add_parser("commands", help="Cetak command eksperimen yang disarankan.")
    # Menyimpan nilai ke `commands.add_argument("--dataset-dir", default` untuk dipakai pada langkah berikutnya.
    commands.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    # Menyimpan nilai ke `commands.add_argument("--json", action` untuk dipakai pada langkah berikutnya.
    commands.add_argument("--json", action="store_true")
    # Menyimpan nilai ke `commands.set_defaults(func` untuk dipakai pada langkah berikutnya.
    commands.set_defaults(func=run_commands)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser


# Membuat langkah kerja bernama `main`.
def main() -> None:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = build_parser()
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parser.parse_args()
    # Melanjutkan langkah kerja pada bagian kode ini.
    args.func(args)


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Melanjutkan langkah kerja pada bagian kode ini.
    main()
