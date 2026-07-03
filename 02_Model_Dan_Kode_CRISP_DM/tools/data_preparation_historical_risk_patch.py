
"""Komentar file skripsi:
Script ringkas fase Data Preparation CRISP-DM untuk ekstraksi mask, risk map, sequence, dan patch sampling.

Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
terutama data understanding, data preparation, modeling, evaluation,
atau deployment sesuai fungsi teknisnya.
"""

from __future__ import annotations

# argparse dipakai agar parameter dataset, patch, threshold, dan output bisa diatur dari command line.
import argparse
# json dipakai untuk menyimpan ringkasan eksperimen, metrik, dan metadata proses.
import json
# re dipakai untuk membaca tanggal dari nama file FIRMS_YYYY-MM-DD.
import re
# asdict dipakai untuk mengubah dataclass record/statistik menjadi JSON laporan.
from dataclasses import asdict, dataclass
# datetime dipakai untuk menjaga urutan kronologis citra dan menghitung target H+1.
from datetime import datetime
# lru_cache dipakai agar mask citra yang sama tidak dibuka dan dihitung ulang berkali-kali.
from functools import lru_cache
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path

# NumPy dipakai untuk mask hotspot, tensor sequence, patch, probability map, dan metrik piksel.
import numpy as np
# Pillow Image dipakai untuk membuka citra hotspot PNG/JPG dan menyimpan preview/hasil.
from PIL import Image, ImageDraw, ImageFilter


# Regex ini mengambil tanggal dari nama file dataset FIRMS agar urutan temporal dapat dibentuk.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")
# Ekstensi default ini membatasi file yang dianggap sebagai citra hotspot valid.
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# Decorator ini membuat record dataset/statistik lebih ringkas tanpa menulis constructor manual.
@dataclass(frozen=True)
# Class ini menyatukan path citra hotspot dan tanggalnya agar urutan temporal tidak tertukar.
class DatasetRecord:
    path: Path
    date: datetime


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


# Membentuk indeks awal sliding window: tujuh citra historis sebagai input dan frame berikutnya sebagai target H+1.
def build_sample_starts(record_count: int, seq_length: int) -> list[int]:
    # Hasil ini dikembalikan sebagai output fungsi `build_sample_starts` untuk tahap berikutnya.
    return list(range(record_count - seq_length))


# Membagi sequence secara kronologis menjadi train, validation, dan test agar tidak ada kebocoran waktu.
def split_sample_starts(
    sample_starts: list[int],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[int], list[int], list[int]]:
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


# Memastikan ukuran kernel dilation/blur bersifat ganjil karena filter gambar membutuhkan pusat kernel.
def _normalize_kernel(size: int) -> int:
    size = max(1, int(size))
    # Hasil ini dikembalikan sebagai output fungsi `_normalize_kernel` untuk tahap berikutnya.
    return size if size % 2 == 1 else size + 1


# Cache ini mempercepat proses karena citra/mask yang sama sering dipakai berulang saat patch sampling.
@lru_cache(maxsize=128)
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
@lru_cache(maxsize=128)
# Membentuk target risk map H+1 dari hotspot merah, dilation label, dan Gaussian blur.
def load_native_risk_map(
    path_str: str,
    native_width: int,
    native_height: int,
    label_dilation_kernel: int,
    label_blur_radius: float,
) -> np.ndarray:
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


# Menghitung ringkasan jumlah patch positif/negatif untuk dilaporkan pada data preparation.
def summarize_patch_entries(
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
) -> dict[str, int]:
    positive_samples = 0
    negative_samples = 0
    # `patch_entry_count` berkaitan dengan potongan citra 160x160 yang dipakai model.
    patch_entry_count = 0

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
            # `patch_entry_count +` berkaitan dengan potongan citra 160x160 yang dipakai model.
            patch_entry_count += positive_patches + negative_patches
        # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
        else:
            negative_samples += 1
            # `patch_entry_count +` berkaitan dengan potongan citra 160x160 yang dipakai model.
            patch_entry_count += negative_patches

    # Hasil ini dikembalikan sebagai output fungsi `summarize_patch_entries` untuk tahap berikutnya.
    return {
        "sample_count": len(sample_starts),
        "positive_sample_count": positive_samples,
        "negative_sample_count": negative_samples,
        "patch_entry_count": patch_entry_count,
        "positive_patches_per_sample": positive_patches,
        "negative_patches_per_sample": negative_patches,
    }


# Mencari sampel demo yang punya input hotspot dan target H+1 agar preview tidak kosong.
def find_demo_sample_start(
    records: list[DatasetRecord],
    train_starts: list[int],
    seq_length: int,
    input_dilation_kernel: int,
    native_width: int,
    native_height: int,
    label_dilation_kernel: int,
    label_blur_radius: float,
    ground_truth_threshold: float,
    sample_mode: str,
) -> int:
    if sample_mode == "first_train":
        # Hasil ini dikembalikan sebagai output fungsi `find_demo_sample_start` untuk tahap berikutnya.
        return train_starts[0]

    for start in train_starts:
        # `input_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
        input_mask = load_native_mask(
            str(records[start + seq_length - 1].path),
            native_width,
            native_height,
            input_dilation_kernel,
        )
        # `target_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
        target_mask = load_native_risk_map(
            str(records[start + seq_length].path),
            native_width,
            native_height,
            label_dilation_kernel,
            label_blur_radius,
        )
        if np.any(input_mask > 0) and np.any(target_mask >= ground_truth_threshold):
            # Hasil ini dikembalikan sebagai output fungsi `find_demo_sample_start` untuk tahap berikutnya.
            return start

    for start in train_starts:
        # `target_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
        target_mask = load_native_risk_map(
            str(records[start + seq_length].path),
            native_width,
            native_height,
            label_dilation_kernel,
            label_blur_radius,
        )
        if np.any(target_mask >= ground_truth_threshold):
            # Hasil ini dikembalikan sebagai output fungsi `find_demo_sample_start` untuk tahap berikutnya.
            return start
    # Hasil ini dikembalikan sebagai output fungsi `find_demo_sample_start` untuk tahap berikutnya.
    return train_starts[0]


# Mengubah array mask/risk map 0-1 menjadi gambar grayscale 0-255 untuk preview laporan.
def array_to_image(array: np.ndarray) -> Image.Image:
    array_2d = array[..., 0] if array.ndim == 3 else array
    # Nilai dikunci ke rentang 0-1 agar cocok sebagai mask/probability map.
    array_uint8 = np.clip(array_2d * 255.0, 0, 255).astype(np.uint8)
    # Hasil ini dikembalikan sebagai output fungsi `array_to_image` untuk tahap berikutnya.
    return Image.fromarray(array_uint8)


# Menandai pusat patch di atas risk map agar strategi sampling patch bisa divisualisasikan.
def build_centers_overlay(base_map: np.ndarray, centers: list[tuple[int, int]], radius: int = 5) -> Image.Image:
    image = array_to_image(base_map).convert("RGB")
    draw = ImageDraw.Draw(image)
    for cy, cx in centers:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 0, 0), width=2)
    # Hasil ini dikembalikan sebagai output fungsi `build_centers_overlay` untuk tahap berikutnya.
    return image


# Menyimpan gambar antara data preparation seperti mask input, risk map, dan contoh patch.
def save_preview_images(
    output_dir: Path,
    input_record: DatasetRecord,
    target_record: DatasetRecord,
    input_mask: np.ndarray,
    risk_map: np.ndarray,
    centers: list[tuple[int, int]],
    mask_patch: np.ndarray,
    risk_patch: np.ndarray,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    with Image.open(input_record.path) as image:
        input_original = image.convert("RGB")
        # `input_original_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
        input_original_path = output_dir / "01_input_original.png"
        input_original.save(input_original_path)
        saved_files.append(str(input_original_path))

    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    with Image.open(target_record.path) as image:
        original = image.convert("RGB")
        # `original_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
        original_path = output_dir / "02_target_original.png"
        original.save(original_path)
        saved_files.append(str(original_path))

    # `mask_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    mask_path = output_dir / "03_input_mask.png"
    array_to_image(input_mask).save(mask_path)
    saved_files.append(str(mask_path))

    # `risk_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    risk_path = output_dir / "04_risk_map.png"
    array_to_image(risk_map).save(risk_path)
    saved_files.append(str(risk_path))

    # `overlay_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    overlay_path = output_dir / "05_risk_map_with_patch_centers.png"
    build_centers_overlay(risk_map, centers).save(overlay_path)
    saved_files.append(str(overlay_path))

    # `mask_patch_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    mask_patch_path = output_dir / "06_example_input_patch.png"
    array_to_image(mask_patch).save(mask_patch_path)
    saved_files.append(str(mask_patch_path))

    # `risk_patch_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    risk_patch_path = output_dir / "07_example_target_patch.png"
    array_to_image(risk_patch).save(risk_patch_path)
    saved_files.append(str(risk_patch_path))

    # Hasil ini dikembalikan sebagai output fungsi `save_preview_images` untuk tahap berikutnya.
    return saved_files


# Menyusun opsi command line agar fase CRISP-DM bisa dijalankan ulang dengan parameter yang jelas.
def build_parser() -> argparse.ArgumentParser:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description=(
            "Demo Data Preparation historical risk patch. "
            "Script ini hanya menjalankan preprocessing inti tanpa training model."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="Ipynb/Dataset History Fire Hotspot In Riau Province PNG",
        help="Folder dataset gambar hotspot historis.",
    )
    # Opsi ini membatasi ekstensi citra yang dibaca dari dataset.
    parser.add_argument("--image-extensions", default=".png")
    # Opsi ini menentukan jumlah frame historis; project memakai 7 frame H-6 sampai H0.
    parser.add_argument("--seq-length", type=int, default=7)
    # Opsi ini mengatur porsi sequence untuk training secara kronologis.
    parser.add_argument("--train-ratio", type=float, default=0.70)
    # Opsi ini mengatur porsi sequence untuk validation setelah training.
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # Opsi ini menyimpan lebar asli peta/citra project.
    parser.add_argument("--native-width", type=int, default=1528)
    # Opsi ini menyimpan tinggi asli peta/citra project.
    parser.add_argument("--native-height", type=int, default=773)
    # Opsi ini mempertebal mask input agar hotspot kecil tidak hilang.
    parser.add_argument("--input-dilation-kernel", type=int, default=5)
    # Opsi ini memperluas label hotspot agar target H+1 tidak terlalu kecil.
    parser.add_argument("--label-dilation-kernel", type=int, default=9)
    # Opsi ini menghaluskan target risk map supaya model belajar area risiko, bukan titik kaku.
    parser.add_argument("--label-blur-radius", type=float, default=2.0)
    # Opsi ini menentukan batas target risk map yang dihitung sebagai positif.
    parser.add_argument("--ground-truth-threshold", type=float, default=0.05)
    # Opsi ini menentukan jumlah patch yang sengaja memuat hotspot.
    parser.add_argument("--positive-patches", type=int, default=4)
    # Opsi ini menentukan jumlah patch background sebagai pembanding.
    parser.add_argument("--negative-patches", type=int, default=1)
    # Opsi ini menentukan lebar patch yang dipotong dari citra besar.
    parser.add_argument("--patch-width", type=int, default=160)
    # Opsi ini menentukan tinggi patch yang dipotong dari citra besar.
    parser.add_argument("--patch-height", type=int, default=160)
    # Opsi `--seed` menambah parameter eksekusi script.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        choices=["auto_positive", "first_train"],
        default="auto_positive",
        help="Pilih sample demo pertama yang mengandung hotspot atau sample train pertama.",
    )
    parser.add_argument(
        "--with-patch-entry-stats",
        action="store_true",
        help="Hitung statistik patch entries seperti pipeline training utama.",
    )
    parser.add_argument(
        "--patch-entry-scope",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Pilih split yang dihitung saat --with-patch-entry-stats dipakai.",
    )
    parser.add_argument(
        "--save-preview-dir",
        default="",
        help="Simpan preview preprocessing ke folder ini. Kosongkan jika tidak perlu.",
    )
    # Opsi ini membuat output ringkasan dicetak sebagai JSON agar mudah dikutip/dibaca ulang.
    parser.add_argument("--json", action="store_true")
    # Hasil ini dikembalikan sebagai output fungsi `build_parser` untuk tahap berikutnya.
    return parser


# Menggabungkan hasil proses menjadi ringkasan JSON/console untuk laporan dan verifikasi eksperimen.
def build_summary(args: argparse.Namespace) -> dict:
    # `dataset_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    # Validasi ini menghentikan proses jika folder dataset hotspot belum tersedia.
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_dir}")

    image_extensions = parse_image_extensions(args.image_extensions)
    # Daftar record citra hotspot yang sudah terbaca dari folder dataset.
    records = load_records(dataset_dir, image_extensions)
    valid_records, skipped_records = validate_records(records)
    # Pengecekan ini memastikan jumlah citra cukup untuk membentuk sequence tujuh hari dan target H+1.
    if len(valid_records) <= args.seq_length:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Jumlah data valid tidak cukup untuk membentuk sequence.")

    # Indeks awal tiap sequence input H-6 sampai H0 untuk memprediksi target H+1.
    sample_starts = build_sample_starts(len(valid_records), args.seq_length)
    train_starts, val_starts, test_starts = split_sample_starts(sample_starts, args.train_ratio, args.val_ratio)

    demo_start = find_demo_sample_start(
        valid_records,
        train_starts,
        args.seq_length,
        args.input_dilation_kernel,
        args.native_width,
        args.native_height,
        args.label_dilation_kernel,
        args.label_blur_radius,
        args.ground_truth_threshold,
        args.sample_mode,
    )
    target_record = valid_records[demo_start + args.seq_length]
    input_record = valid_records[demo_start + args.seq_length - 1]

    # `input_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
    input_mask = load_native_mask(
        str(input_record.path),
        args.native_width,
        args.native_height,
        args.input_dilation_kernel,
    )
    # `risk_map` menyimpan peta risiko/target yang dipelajari atau divisualisasikan.
    risk_map = load_native_risk_map(
        str(target_record.path),
        args.native_width,
        args.native_height,
        args.label_dilation_kernel,
        args.label_blur_radius,
    )

    rng = np.random.default_rng(args.seed)
    # Daftar pusat patch yang akan dipotong dari citra besar.
    centers = sample_patch_centers(
        risk_map,
        # `positive_patch_count` berkaitan dengan potongan citra 160x160 yang dipakai model.
        positive_patch_count=args.positive_patches,
        # `negative_patch_count` berkaitan dengan potongan citra 160x160 yang dipakai model.
        negative_patch_count=args.negative_patches,
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        ground_truth_threshold=args.ground_truth_threshold,
        rng=rng,
    )
    first_cy, first_cx = centers[0]
    # `mask_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
    mask_patch = extract_patch(input_mask, first_cy, first_cx, args.patch_height, args.patch_width)
    # `risk_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
    risk_patch = extract_patch(risk_map, first_cy, first_cx, args.patch_height, args.patch_width)

    # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
    summary: dict = {
        "dataset_dir": str(dataset_dir),
        "record_count_total": len(records),
        "record_count_valid": len(valid_records),
        "record_count_skipped": len(skipped_records),
        "train_samples": len(train_starts),
        "val_samples": len(val_starts),
        "test_samples": len(test_starts),
        "demo_sample_start_index": demo_start,
        "demo_input_date": input_record.date.date().isoformat(),
        "demo_target_date": target_record.date.date().isoformat(),
        "input_mask_shape": list(input_mask.shape),
        "risk_map_shape": list(risk_map.shape),
        # Jumlah piksel positif dihitung untuk menunjukkan kelangkaan hotspot pada laporan.
        "input_mask_positive_pixels": int(np.count_nonzero(input_mask > 0)),
        # `"risk_map_positive_pixels_thresholded"` berkaitan dengan ambang untuk membedakan area risiko dan background.
        "risk_map_positive_pixels_thresholded": int(np.count_nonzero(risk_map >= args.ground_truth_threshold)),
        "sampled_patch_center_count": len(centers),
        "sampled_patch_centers_preview": [[int(cy), int(cx)] for cy, cx in centers[:10]],
        "first_input_patch_shape": list(mask_patch.shape),
        "first_target_patch_shape": list(risk_patch.shape),
        "input_record": asdict(input_record) | {"path": str(input_record.path)},
        "target_record": asdict(target_record) | {"path": str(target_record.path)},
        "patch_settings": {
            "patch_width": args.patch_width,
            "patch_height": args.patch_height,
            "positive_patches": args.positive_patches,
            "negative_patches": args.negative_patches,
            "ground_truth_threshold": args.ground_truth_threshold,
        },
        "preprocessing_settings": {
            "input_dilation_kernel": args.input_dilation_kernel,
            "label_dilation_kernel": args.label_dilation_kernel,
            "label_blur_radius": args.label_blur_radius,
        },
        "patch_entry_stats": None,
        "saved_preview_files": [],
    }

    if args.with_patch_entry_stats:
        # `patch_stats` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_stats: dict[str, dict[str, int]] = {}
        if args.patch_entry_scope in {"train", "all"}:
            # `patch_stats["train"]` berkaitan dengan potongan citra 160x160 yang dipakai model.
            patch_stats["train"] = summarize_patch_entries(
                valid_records,
                train_starts,
                seq_length=args.seq_length,
                native_width=args.native_width,
                native_height=args.native_height,
                label_dilation_kernel=args.label_dilation_kernel,
                label_blur_radius=args.label_blur_radius,
                # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
                ground_truth_threshold=args.ground_truth_threshold,
                # `positive_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
                positive_patches=args.positive_patches,
                # `negative_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
                negative_patches=args.negative_patches,
            )
        if args.patch_entry_scope in {"val", "all"}:
            # `patch_stats["val"]` berkaitan dengan potongan citra 160x160 yang dipakai model.
            patch_stats["val"] = summarize_patch_entries(
                valid_records,
                val_starts,
                seq_length=args.seq_length,
                native_width=args.native_width,
                native_height=args.native_height,
                label_dilation_kernel=args.label_dilation_kernel,
                label_blur_radius=args.label_blur_radius,
                # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
                ground_truth_threshold=args.ground_truth_threshold,
                # `positive_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
                positive_patches=1,
                # `negative_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
                negative_patches=1,
            )
        if args.patch_entry_scope in {"test", "all"}:
            # `patch_stats["test"]` berkaitan dengan potongan citra 160x160 yang dipakai model.
            patch_stats["test"] = summarize_patch_entries(
                valid_records,
                test_starts,
                seq_length=args.seq_length,
                native_width=args.native_width,
                native_height=args.native_height,
                label_dilation_kernel=args.label_dilation_kernel,
                label_blur_radius=args.label_blur_radius,
                # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
                ground_truth_threshold=args.ground_truth_threshold,
                # `positive_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
                positive_patches=1,
                # `negative_patches` berkaitan dengan potongan citra 160x160 yang dipakai model.
                negative_patches=1,
            )
        # `summary["patch_entry_stats"]` berkaitan dengan potongan citra 160x160 yang dipakai model.
        summary["patch_entry_stats"] = patch_stats

    if args.save_preview_dir:
        saved_files = save_preview_images(
            # `output_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
            output_dir=Path(args.save_preview_dir).expanduser().resolve(),
            input_record=input_record,
            target_record=target_record,
            # `input_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
            input_mask=input_mask,
            # `risk_map` menyimpan peta risiko/target yang dipelajari atau divisualisasikan.
            risk_map=risk_map,
            # Daftar pusat patch yang akan dipotong dari citra besar.
            centers=centers,
            # `mask_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
            mask_patch=mask_patch,
            # `risk_patch` berkaitan dengan potongan citra 160x160 yang dipakai model.
            risk_patch=risk_patch,
        )
        summary["saved_preview_files"] = saved_files

    # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
    return summary


# Fungsi `print_human_summary` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def print_human_summary(summary: dict) -> None:
    print("[data_preparation] Dataset valid:", summary["record_count_valid"])
    print(
        "[data_preparation] Demo sample:",
        f"input {summary['demo_input_date']} -> target {summary['demo_target_date']}",
    )
    print("[data_preparation] Input mask shape:", tuple(summary["input_mask_shape"]))
    print("[data_preparation] Risk map shape:", tuple(summary["risk_map_shape"]))
    print("[data_preparation] Input mask positive pixels:", summary["input_mask_positive_pixels"])
    print(
        "[data_preparation] Risk map positive pixels (thresholded):",
        summary["risk_map_positive_pixels_thresholded"],
    )
    print("[data_preparation] Sampled patch centers:", summary["sampled_patch_center_count"])
    print("[data_preparation] Centers preview:", summary["sampled_patch_centers_preview"])
    print("[data_preparation] Example input patch shape:", tuple(summary["first_input_patch_shape"]))
    print("[data_preparation] Example target patch shape:", tuple(summary["first_target_patch_shape"]))

    # `patch_stats` berkaitan dengan potongan citra 160x160 yang dipakai model.
    patch_stats = summary["patch_entry_stats"]
    if patch_stats is not None:
        if "train" in patch_stats:
            print("[data_preparation] Train patch entries:", patch_stats["train"]["patch_entry_count"])
        if "val" in patch_stats:
            print("[data_preparation] Val patch entries:", patch_stats["val"]["patch_entry_count"])
        if "test" in patch_stats:
            print("[data_preparation] Test patch entries:", patch_stats["test"]["patch_entry_count"])
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        print("[data_preparation] Patch entry stats: dilewati")
        print("[data_preparation] Tips: pakai --with-patch-entry-stats untuk menghitung statistik penuh.")

    if summary["saved_preview_files"]:
        print("[data_preparation] Preview files:")
        for path in summary["saved_preview_files"]:
            print("-", path)


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> None:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = build_parser()
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parser.parse_args()
    # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
    summary = build_summary(args)
    print_human_summary(summary)
    if args.json:
        print()
        # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
