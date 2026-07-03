# File anotasi dari `tools/data_preparation_historical_risk_patch.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Data Preparation, yaitu menyiapkan citra menjadi mask, sequence, dan patch.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Script ringkas fase Data Preparation CRISP-DM untuk ekstraksi mask, risk map, sequence, dan patch sampling.

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
from dataclasses import asdict, dataclass
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from datetime import datetime
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from functools import lru_cache
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image, ImageDraw, ImageFilter


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


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


# Membuat langkah kerja bernama `build_sample_starts`.
def build_sample_starts(record_count: int, seq_length: int) -> list[int]:
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
        raise ValueError("Split train/val/test gagal. Periksa rasio dataset.")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return train, val, test


# Membuat langkah kerja bernama `_normalize_kernel`.
def _normalize_kernel(size: int) -> int:
    # Menyimpan nilai ke `size` untuk dipakai pada langkah berikutnya.
    size = max(1, int(size))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return size if size % 2 == 1 else size + 1


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@lru_cache(maxsize=128)
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
@lru_cache(maxsize=128)
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


# Membuat langkah kerja bernama `summarize_patch_entries`.
def summarize_patch_entries(
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
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> dict[str, int]:
    # Menyimpan nilai ke `positive_samples` untuk dipakai pada langkah berikutnya.
    positive_samples = 0
    # Menyimpan nilai ke `negative_samples` untuk dipakai pada langkah berikutnya.
    negative_samples = 0
    # Menyimpan nilai ke `patch_entry_count` untuk dipakai pada langkah berikutnya.
    patch_entry_count = 0

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
            # Menyimpan nilai ke `patch_entry_count +` untuk dipakai pada langkah berikutnya.
            patch_entry_count += positive_patches + negative_patches
        # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
        else:
            # Menyimpan nilai ke `negative_samples +` untuk dipakai pada langkah berikutnya.
            negative_samples += 1
            # Menyimpan nilai ke `patch_entry_count +` untuk dipakai pada langkah berikutnya.
            patch_entry_count += negative_patches

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sample_count": len(sample_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_sample_count": positive_samples,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "negative_sample_count": negative_samples,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_entry_count": patch_entry_count,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_patches_per_sample": positive_patches,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "negative_patches_per_sample": negative_patches,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }


# Membuat langkah kerja bernama `find_demo_sample_start`.
def find_demo_sample_start(
    # Menjelaskan data `records` yang disimpan atau dikirim pada bagian ini.
    records: list[DatasetRecord],
    # Menjelaskan data `train_starts` yang disimpan atau dikirim pada bagian ini.
    train_starts: list[int],
    # Menjelaskan data `seq_length` yang disimpan atau dikirim pada bagian ini.
    seq_length: int,
    # Menjelaskan data `input_dilation_kernel` yang disimpan atau dikirim pada bagian ini.
    input_dilation_kernel: int,
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
    # Menjelaskan data `sample_mode` yang disimpan atau dikirim pada bagian ini.
    sample_mode: str,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> int:
    # Mengecek syarat sebelum melanjutkan proses.
    if sample_mode == "first_train":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return train_starts[0]

    # Mengulang proses untuk setiap data dalam daftar.
    for start in train_starts:
        # Menyimpan nilai ke `input_mask` untuk dipakai pada langkah berikutnya.
        input_mask = load_native_mask(
            # Melanjutkan langkah kerja pada bagian kode ini.
            str(records[start + seq_length - 1].path),
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            native_width,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            native_height,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            input_dilation_kernel,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
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
        # Mengecek syarat sebelum melanjutkan proses.
        if np.any(input_mask > 0) and np.any(target_mask >= ground_truth_threshold):
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return start

    # Mengulang proses untuk setiap data dalam daftar.
    for start in train_starts:
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
        # Mengecek syarat sebelum melanjutkan proses.
        if np.any(target_mask >= ground_truth_threshold):
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return start
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return train_starts[0]


# Membuat langkah kerja bernama `array_to_image`.
def array_to_image(array: np.ndarray) -> Image.Image:
    # Menyimpan nilai ke `array_2d` untuk dipakai pada langkah berikutnya.
    array_2d = array[..., 0] if array.ndim == 3 else array
    # Menyimpan nilai ke `array_uint8` untuk dipakai pada langkah berikutnya.
    array_uint8 = np.clip(array_2d * 255.0, 0, 255).astype(np.uint8)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return Image.fromarray(array_uint8)


# Membuat langkah kerja bernama `build_centers_overlay`.
def build_centers_overlay(base_map: np.ndarray, centers: list[tuple[int, int]], radius: int = 5) -> Image.Image:
    # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
    image = array_to_image(base_map).convert("RGB")
    # Menyimpan nilai ke `draw` untuk dipakai pada langkah berikutnya.
    draw = ImageDraw.Draw(image)
    # Mengulang proses untuk setiap data dalam daftar.
    for cy, cx in centers:
        # Menyimpan nilai ke `draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline` untuk dipakai pada langkah berikutnya.
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 0, 0), width=2)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return image


# Membuat langkah kerja bernama `save_preview_images`.
def save_preview_images(
    # Menjelaskan data `output_dir` yang disimpan atau dikirim pada bagian ini.
    output_dir: Path,
    # Menjelaskan data `input_record` yang disimpan atau dikirim pada bagian ini.
    input_record: DatasetRecord,
    # Menjelaskan data `target_record` yang disimpan atau dikirim pada bagian ini.
    target_record: DatasetRecord,
    # Menjelaskan data `input_mask` yang disimpan atau dikirim pada bagian ini.
    input_mask: np.ndarray,
    # Menjelaskan data `risk_map` yang disimpan atau dikirim pada bagian ini.
    risk_map: np.ndarray,
    # Menjelaskan data `centers` yang disimpan atau dikirim pada bagian ini.
    centers: list[tuple[int, int]],
    # Menjelaskan data `mask_patch` yang disimpan atau dikirim pada bagian ini.
    mask_patch: np.ndarray,
    # Menjelaskan data `risk_patch` yang disimpan atau dikirim pada bagian ini.
    risk_patch: np.ndarray,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> list[str]:
    # Menyimpan nilai ke `output_dir.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `saved_files` untuk dipakai pada langkah berikutnya.
    saved_files: list[str] = []

    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with Image.open(input_record.path) as image:
        # Menyimpan nilai ke `input_original` untuk dipakai pada langkah berikutnya.
        input_original = image.convert("RGB")
        # Menyimpan nilai ke `input_original_path` untuk dipakai pada langkah berikutnya.
        input_original_path = output_dir / "01_input_original.png"
        # Melanjutkan langkah kerja pada bagian kode ini.
        input_original.save(input_original_path)
        # Melanjutkan langkah kerja pada bagian kode ini.
        saved_files.append(str(input_original_path))

    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with Image.open(target_record.path) as image:
        # Menyimpan nilai ke `original` untuk dipakai pada langkah berikutnya.
        original = image.convert("RGB")
        # Menyimpan nilai ke `original_path` untuk dipakai pada langkah berikutnya.
        original_path = output_dir / "02_target_original.png"
        # Melanjutkan langkah kerja pada bagian kode ini.
        original.save(original_path)
        # Melanjutkan langkah kerja pada bagian kode ini.
        saved_files.append(str(original_path))

    # Menyimpan nilai ke `mask_path` untuk dipakai pada langkah berikutnya.
    mask_path = output_dir / "03_input_mask.png"
    # Melanjutkan langkah kerja pada bagian kode ini.
    array_to_image(input_mask).save(mask_path)
    # Melanjutkan langkah kerja pada bagian kode ini.
    saved_files.append(str(mask_path))

    # Menyimpan nilai ke `risk_path` untuk dipakai pada langkah berikutnya.
    risk_path = output_dir / "04_risk_map.png"
    # Melanjutkan langkah kerja pada bagian kode ini.
    array_to_image(risk_map).save(risk_path)
    # Melanjutkan langkah kerja pada bagian kode ini.
    saved_files.append(str(risk_path))

    # Menyimpan nilai ke `overlay_path` untuk dipakai pada langkah berikutnya.
    overlay_path = output_dir / "05_risk_map_with_patch_centers.png"
    # Melanjutkan langkah kerja pada bagian kode ini.
    build_centers_overlay(risk_map, centers).save(overlay_path)
    # Melanjutkan langkah kerja pada bagian kode ini.
    saved_files.append(str(overlay_path))

    # Menyimpan nilai ke `mask_patch_path` untuk dipakai pada langkah berikutnya.
    mask_patch_path = output_dir / "06_example_input_patch.png"
    # Melanjutkan langkah kerja pada bagian kode ini.
    array_to_image(mask_patch).save(mask_patch_path)
    # Melanjutkan langkah kerja pada bagian kode ini.
    saved_files.append(str(mask_patch_path))

    # Menyimpan nilai ke `risk_patch_path` untuk dipakai pada langkah berikutnya.
    risk_patch_path = output_dir / "07_example_target_patch.png"
    # Melanjutkan langkah kerja pada bagian kode ini.
    array_to_image(risk_patch).save(risk_patch_path)
    # Melanjutkan langkah kerja pada bagian kode ini.
    saved_files.append(str(risk_patch_path))

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return saved_files


# Membuat langkah kerja bernama `build_parser`.
def build_parser() -> argparse.ArgumentParser:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description=(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Demo Data Preparation historical risk patch. "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Script ini hanya menjalankan preprocessing inti tanpa training model."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--dataset-dir",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="Ipynb/Dataset History Fire Hotspot In Riau Province PNG",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Folder dataset gambar hotspot historis.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--image-extensions", default=".png")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--seq-length", type=int, default=7)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--train-ratio", type=float, default=0.70)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--native-width", type=int, default=1528)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--native-height", type=int, default=773)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--input-dilation-kernel", type=int, default=5)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--label-dilation-kernel", type=int, default=9)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--label-blur-radius", type=float, default=2.0)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--ground-truth-threshold", type=float, default=0.05)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--positive-patches", type=int, default=4)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--negative-patches", type=int, default=1)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--patch-width", type=int, default=160)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--patch-height", type=int, default=160)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--seed", type=int, default=42)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--sample-mode",
        # Menyimpan nilai ke `choices` untuk dipakai pada langkah berikutnya.
        choices=["auto_positive", "first_train"],
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="auto_positive",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Pilih sample demo pertama yang mengandung hotspot atau sample train pertama.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--with-patch-entry-stats",
        # Menyimpan nilai ke `action` untuk dipakai pada langkah berikutnya.
        action="store_true",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Hitung statistik patch entries seperti pipeline training utama.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--patch-entry-scope",
        # Menyimpan nilai ke `choices` untuk dipakai pada langkah berikutnya.
        choices=["train", "val", "test", "all"],
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="train",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Pilih split yang dihitung saat --with-patch-entry-stats dipakai.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--save-preview-dir",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Simpan preview preprocessing ke folder ini. Kosongkan jika tidak perlu.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--json", action="store_true")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser


# Membuat langkah kerja bernama `build_summary`.
def build_summary(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `dataset_dir` untuk dipakai pada langkah berikutnya.
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    # Mengecek syarat sebelum melanjutkan proses.
    if not dataset_dir.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_dir}")

    # Menyimpan nilai ke `image_extensions` untuk dipakai pada langkah berikutnya.
    image_extensions = parse_image_extensions(args.image_extensions)
    # Menyimpan nilai ke `records` untuk dipakai pada langkah berikutnya.
    records = load_records(dataset_dir, image_extensions)
    # Menyimpan nilai ke `valid_records, skipped_records` untuk dipakai pada langkah berikutnya.
    valid_records, skipped_records = validate_records(records)
    # Mengecek syarat sebelum melanjutkan proses.
    if len(valid_records) <= args.seq_length:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Jumlah data valid tidak cukup untuk membentuk sequence.")

    # Menyimpan nilai ke `sample_starts` untuk dipakai pada langkah berikutnya.
    sample_starts = build_sample_starts(len(valid_records), args.seq_length)
    # Menyimpan nilai ke `train_starts, val_starts, test_starts` untuk dipakai pada langkah berikutnya.
    train_starts, val_starts, test_starts = split_sample_starts(sample_starts, args.train_ratio, args.val_ratio)

    # Menyimpan nilai ke `demo_start` untuk dipakai pada langkah berikutnya.
    demo_start = find_demo_sample_start(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        valid_records,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        train_starts,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.seq_length,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.input_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.native_width,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.native_height,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.label_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.label_blur_radius,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.ground_truth_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.sample_mode,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `target_record` untuk dipakai pada langkah berikutnya.
    target_record = valid_records[demo_start + args.seq_length]
    # Menyimpan nilai ke `input_record` untuk dipakai pada langkah berikutnya.
    input_record = valid_records[demo_start + args.seq_length - 1]

    # Menyimpan nilai ke `input_mask` untuk dipakai pada langkah berikutnya.
    input_mask = load_native_mask(
        # Melanjutkan langkah kerja pada bagian kode ini.
        str(input_record.path),
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.native_width,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.native_height,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.input_dilation_kernel,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `risk_map` untuk dipakai pada langkah berikutnya.
    risk_map = load_native_risk_map(
        # Melanjutkan langkah kerja pada bagian kode ini.
        str(target_record.path),
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.native_width,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.native_height,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.label_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.label_blur_radius,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `rng` untuk dipakai pada langkah berikutnya.
    rng = np.random.default_rng(args.seed)
    # Menyimpan nilai ke `centers` untuk dipakai pada langkah berikutnya.
    centers = sample_patch_centers(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        risk_map,
        # Menyimpan nilai ke `positive_patch_count` untuk dipakai pada langkah berikutnya.
        positive_patch_count=args.positive_patches,
        # Menyimpan nilai ke `negative_patch_count` untuk dipakai pada langkah berikutnya.
        negative_patch_count=args.negative_patches,
        # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        ground_truth_threshold=args.ground_truth_threshold,
        # Menyimpan nilai ke `rng` untuk dipakai pada langkah berikutnya.
        rng=rng,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `first_cy, first_cx` untuk dipakai pada langkah berikutnya.
    first_cy, first_cx = centers[0]
    # Menyimpan nilai ke `mask_patch` untuk dipakai pada langkah berikutnya.
    mask_patch = extract_patch(input_mask, first_cy, first_cx, args.patch_height, args.patch_width)
    # Menyimpan nilai ke `risk_patch` untuk dipakai pada langkah berikutnya.
    risk_patch = extract_patch(risk_map, first_cy, first_cx, args.patch_height, args.patch_width)

    # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
    summary: dict = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dataset_dir": str(dataset_dir),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "record_count_total": len(records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "record_count_valid": len(valid_records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "record_count_skipped": len(skipped_records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_samples": len(train_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_samples": len(val_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_samples": len(test_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "demo_sample_start_index": demo_start,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "demo_input_date": input_record.date.date().isoformat(),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "demo_target_date": target_record.date.date().isoformat(),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_mask_shape": list(input_mask.shape),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "risk_map_shape": list(risk_map.shape),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_mask_positive_pixels": int(np.count_nonzero(input_mask > 0)),
        # Menyimpan nilai ke `"risk_map_positive_pixels_thresholded"` untuk dipakai pada langkah berikutnya.
        "risk_map_positive_pixels_thresholded": int(np.count_nonzero(risk_map >= args.ground_truth_threshold)),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sampled_patch_center_count": len(centers),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sampled_patch_centers_preview": [[int(cy), int(cx)] for cy, cx in centers[:10]],
        # Melanjutkan langkah kerja pada bagian kode ini.
        "first_input_patch_shape": list(mask_patch.shape),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "first_target_patch_shape": list(risk_patch.shape),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_record": asdict(input_record) | {"path": str(input_record.path)},
        # Melanjutkan langkah kerja pada bagian kode ini.
        "target_record": asdict(target_record) | {"path": str(target_record.path)},
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_settings": {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "patch_width": args.patch_width,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "patch_height": args.patch_height,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "positive_patches": args.positive_patches,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "negative_patches": args.negative_patches,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "ground_truth_threshold": args.ground_truth_threshold,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Melanjutkan langkah kerja pada bagian kode ini.
        "preprocessing_settings": {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "input_dilation_kernel": args.input_dilation_kernel,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "label_dilation_kernel": args.label_dilation_kernel,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "label_blur_radius": args.label_blur_radius,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_entry_stats": None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "saved_preview_files": [],
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }

    # Mengecek syarat sebelum melanjutkan proses.
    if args.with_patch_entry_stats:
        # Menyimpan nilai ke `patch_stats` untuk dipakai pada langkah berikutnya.
        patch_stats: dict[str, dict[str, int]] = {}
        # Mengecek syarat sebelum melanjutkan proses.
        if args.patch_entry_scope in {"train", "all"}:
            # Menyimpan nilai ke `patch_stats["train"]` untuk dipakai pada langkah berikutnya.
            patch_stats["train"] = summarize_patch_entries(
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                valid_records,
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
                positive_patches=args.positive_patches,
                # Menyimpan nilai ke `negative_patches` untuk dipakai pada langkah berikutnya.
                negative_patches=args.negative_patches,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
        # Mengecek syarat sebelum melanjutkan proses.
        if args.patch_entry_scope in {"val", "all"}:
            # Menyimpan nilai ke `patch_stats["val"]` untuk dipakai pada langkah berikutnya.
            patch_stats["val"] = summarize_patch_entries(
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                valid_records,
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
                positive_patches=1,
                # Menyimpan nilai ke `negative_patches` untuk dipakai pada langkah berikutnya.
                negative_patches=1,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
        # Mengecek syarat sebelum melanjutkan proses.
        if args.patch_entry_scope in {"test", "all"}:
            # Menyimpan nilai ke `patch_stats["test"]` untuk dipakai pada langkah berikutnya.
            patch_stats["test"] = summarize_patch_entries(
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                valid_records,
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
                positive_patches=1,
                # Menyimpan nilai ke `negative_patches` untuk dipakai pada langkah berikutnya.
                negative_patches=1,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
        # Menyimpan nilai ke `summary["patch_entry_stats"]` untuk dipakai pada langkah berikutnya.
        summary["patch_entry_stats"] = patch_stats

    # Mengecek syarat sebelum melanjutkan proses.
    if args.save_preview_dir:
        # Menyimpan nilai ke `saved_files` untuk dipakai pada langkah berikutnya.
        saved_files = save_preview_images(
            # Menyimpan nilai ke `output_dir` untuk dipakai pada langkah berikutnya.
            output_dir=Path(args.save_preview_dir).expanduser().resolve(),
            # Menyimpan nilai ke `input_record` untuk dipakai pada langkah berikutnya.
            input_record=input_record,
            # Menyimpan nilai ke `target_record` untuk dipakai pada langkah berikutnya.
            target_record=target_record,
            # Menyimpan nilai ke `input_mask` untuk dipakai pada langkah berikutnya.
            input_mask=input_mask,
            # Menyimpan nilai ke `risk_map` untuk dipakai pada langkah berikutnya.
            risk_map=risk_map,
            # Menyimpan nilai ke `centers` untuk dipakai pada langkah berikutnya.
            centers=centers,
            # Menyimpan nilai ke `mask_patch` untuk dipakai pada langkah berikutnya.
            mask_patch=mask_patch,
            # Menyimpan nilai ke `risk_patch` untuk dipakai pada langkah berikutnya.
            risk_patch=risk_patch,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `summary["saved_preview_files"]` untuk dipakai pada langkah berikutnya.
        summary["saved_preview_files"] = saved_files

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return summary


# Membuat langkah kerja bernama `print_human_summary`.
def print_human_summary(summary: dict) -> None:
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_preparation] Dataset valid:", summary["record_count_valid"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "[data_preparation] Demo sample:",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"input {summary['demo_input_date']} -> target {summary['demo_target_date']}",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_preparation] Input mask shape:", tuple(summary["input_mask_shape"]))
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_preparation] Risk map shape:", tuple(summary["risk_map_shape"]))
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_preparation] Input mask positive pixels:", summary["input_mask_positive_pixels"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "[data_preparation] Risk map positive pixels (thresholded):",
        # Melanjutkan langkah kerja pada bagian kode ini.
        summary["risk_map_positive_pixels_thresholded"],
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_preparation] Sampled patch centers:", summary["sampled_patch_center_count"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_preparation] Centers preview:", summary["sampled_patch_centers_preview"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_preparation] Example input patch shape:", tuple(summary["first_input_patch_shape"]))
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_preparation] Example target patch shape:", tuple(summary["first_target_patch_shape"]))

    # Menyimpan nilai ke `patch_stats` untuk dipakai pada langkah berikutnya.
    patch_stats = summary["patch_entry_stats"]
    # Mengecek syarat sebelum melanjutkan proses.
    if patch_stats is not None:
        # Mengecek syarat sebelum melanjutkan proses.
        if "train" in patch_stats:
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print("[data_preparation] Train patch entries:", patch_stats["train"]["patch_entry_count"])
        # Mengecek syarat sebelum melanjutkan proses.
        if "val" in patch_stats:
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print("[data_preparation] Val patch entries:", patch_stats["val"]["patch_entry_count"])
        # Mengecek syarat sebelum melanjutkan proses.
        if "test" in patch_stats:
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print("[data_preparation] Test patch entries:", patch_stats["test"]["patch_entry_count"])
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[data_preparation] Patch entry stats: dilewati")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[data_preparation] Tips: pakai --with-patch-entry-stats untuk menghitung statistik penuh.")

    # Mengecek syarat sebelum melanjutkan proses.
    if summary["saved_preview_files"]:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[data_preparation] Preview files:")
        # Mengulang proses untuk setiap data dalam daftar.
        for path in summary["saved_preview_files"]:
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print("-", path)


# Membuat langkah kerja bernama `main`.
def main() -> None:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = build_parser()
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parser.parse_args()
    # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
    summary = build_summary(args)
    # Melanjutkan langkah kerja pada bagian kode ini.
    print_human_summary(summary)
    # Mengecek syarat sebelum melanjutkan proses.
    if args.json:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print()
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(json.dumps(summary, indent=2, default=str))


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Melanjutkan langkah kerja pada bagian kode ini.
    main()
