# File anotasi dari `tools/data_understanding_historical_risk_patch.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Data Understanding, yaitu memahami dataset citra hotspot.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Script ringkas fase Data Understanding CRISP-DM untuk membaca dataset, sequence, split kronologis, dan class imbalance.

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
from PIL import Image, ImageFilter


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


# Membuat langkah kerja bernama `build_parser`.
def build_parser() -> argparse.ArgumentParser:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description=(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Ringkasan Data Understanding untuk historical risk patch. "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Script ini hanya membaca dataset, split kronologis, dan opsional menghitung class imbalance."
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
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--image-extensions",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=".png",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Daftar ekstensi gambar, pisahkan dengan koma. Contoh: .jpg,.jpeg,.png",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
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
    parser.add_argument("--label-dilation-kernel", type=int, default=9)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--label-blur-radius", type=float, default=2.0)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--with-positive-weight",
        # Menyimpan nilai ke `action` untuk dipakai pada langkah berikutnya.
        action="store_true",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Hitung positive mass ratio dan positive class weight seperti training script utama.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--show-skipped",
        # Menyimpan nilai ke `action` untuk dipakai pada langkah berikutnya.
        action="store_true",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Tampilkan daftar file yang dilewati saat validasi.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--json",
        # Menyimpan nilai ke `action` untuk dipakai pada langkah berikutnya.
        action="store_true",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Cetak ringkasan akhir dalam format JSON.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
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
    # Mengecek syarat sebelum melanjutkan proses.
    if len(records) <= args.seq_length:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Jumlah data tidak cukup untuk membentuk sequence.")

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
    # Menyimpan nilai ke `pos_weight` untuk dipakai pada langkah berikutnya.
    pos_weight: float | None = None
    # Menyimpan nilai ke `positive_ratio` untuk dipakai pada langkah berikutnya.
    positive_ratio: float | None = None
    # Mengecek syarat sebelum melanjutkan proses.
    if args.with_positive_weight:
        # Menyimpan nilai ke `pos_weight, positive_ratio` untuk dipakai pada langkah berikutnya.
        pos_weight, positive_ratio = estimate_positive_weight(
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            valid_records,
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

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dataset_dir": str(dataset_dir),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "image_extensions": list(image_extensions),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "record_count_total": len(records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "record_count_valid": len(valid_records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "record_count_skipped": len(skipped_records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "date_start": valid_records[0].date.date().isoformat(),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "date_end": valid_records[-1].date.date().isoformat(),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "seq_length": args.seq_length,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sample_count_total": len(sample_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_samples": len(train_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_samples": len(val_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_samples": len(test_starts),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_mass_ratio_train": positive_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_class_weight": pos_weight,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_weight_computed": args.with_positive_weight,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_ratio": args.train_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_ratio": args.val_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_ratio_effective": len(test_starts) / max(len(sample_starts), 1),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "skipped_records_preview": skipped_records[:10],
        # Melanjutkan langkah kerja pada bagian kode ini.
        "first_record": asdict(valid_records[0]) | {"path": str(valid_records[0].path)},
        # Melanjutkan langkah kerja pada bagian kode ini.
        "last_record": asdict(valid_records[-1]) | {"path": str(valid_records[-1].path)},
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }


# Membuat langkah kerja bernama `print_human_summary`.
def print_human_summary(summary: dict, show_skipped: bool) -> None:
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_understanding] Dataset ditemukan:", f"{summary['record_count_valid']} frame valid")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_understanding] Total file cocok ekstensi:", summary["record_count_total"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_understanding] Rentang data:", f"{summary['date_start']} s.d. {summary['date_end']}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_understanding] Seq length:", summary["seq_length"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_understanding] Total sample sequence:", summary["sample_count_total"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_understanding] Train samples:", summary["train_samples"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_understanding] Val samples:", summary["val_samples"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[data_understanding] Test samples:", summary["test_samples"])
    # Mengecek syarat sebelum melanjutkan proses.
    if summary["positive_weight_computed"]:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "[data_understanding] Positive mass ratio (train):",
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{summary['positive_mass_ratio_train']:.8f}",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "[data_understanding] Positive class weight:",
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{summary['positive_class_weight']:.4f}",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[data_understanding] Positive mass ratio (train): dilewati")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[data_understanding] Positive class weight: dilewati")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[data_understanding] Tips: pakai --with-positive-weight untuk menghitung nilai exact.")

    # Mengecek syarat sebelum melanjutkan proses.
    if summary["record_count_skipped"] > 0:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[data_understanding] File dilewati:", summary["record_count_skipped"])
        # Mengecek syarat sebelum melanjutkan proses.
        if show_skipped:
            # Mengulang proses untuk setiap data dalam daftar.
            for item in summary["skipped_records_preview"]:
                # Menampilkan informasi ke terminal agar proses mudah dicek.
                print(f"- skip {item['path']} | alasan: {item['reason']}")


# Membuat langkah kerja bernama `main`.
def main() -> None:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = build_parser()
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parser.parse_args()
    # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
    summary = build_summary(args)
    # Menyimpan nilai ke `print_human_summary(summary, show_skipped` untuk dipakai pada langkah berikutnya.
    print_human_summary(summary, show_skipped=args.show_skipped)
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
