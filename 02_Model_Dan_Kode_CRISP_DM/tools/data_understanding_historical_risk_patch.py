
"""Komentar file skripsi:
Script ringkas fase Data Understanding CRISP-DM untuk membaca dataset, sequence, split kronologis, dan class imbalance.

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
from PIL import Image, ImageFilter


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


# Menyusun opsi command line agar fase CRISP-DM bisa dijalankan ulang dengan parameter yang jelas.
def build_parser() -> argparse.ArgumentParser:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description=(
            "Ringkasan Data Understanding untuk historical risk patch. "
            "Script ini hanya membaca dataset, split kronologis, dan opsional menghitung class imbalance."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="Ipynb/Dataset History Fire Hotspot In Riau Province PNG",
        help="Folder dataset gambar hotspot historis.",
    )
    parser.add_argument(
        "--image-extensions",
        default=".png",
        help="Daftar ekstensi gambar, pisahkan dengan koma. Contoh: .jpg,.jpeg,.png",
    )
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
    # Opsi ini memperluas label hotspot agar target H+1 tidak terlalu kecil.
    parser.add_argument("--label-dilation-kernel", type=int, default=9)
    # Opsi ini menghaluskan target risk map supaya model belajar area risiko, bukan titik kaku.
    parser.add_argument("--label-blur-radius", type=float, default=2.0)
    # Opsi `--max-pos-weight` menambah parameter eksekusi script.
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument(
        "--with-positive-weight",
        action="store_true",
        help="Hitung positive mass ratio dan positive class weight seperti training script utama.",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Tampilkan daftar file yang dilewati saat validasi.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Cetak ringkasan akhir dalam format JSON.",
    )
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
    # Pengecekan ini memastikan jumlah citra cukup untuk membentuk sequence tujuh hari dan target H+1.
    if len(records) <= args.seq_length:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Jumlah data tidak cukup untuk membentuk sequence.")

    valid_records, skipped_records = validate_records(records)
    # Pengecekan ini memastikan jumlah citra cukup untuk membentuk sequence tujuh hari dan target H+1.
    if len(valid_records) <= args.seq_length:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Jumlah data valid tidak cukup untuk membentuk sequence.")

    # Indeks awal tiap sequence input H-6 sampai H0 untuk memprediksi target H+1.
    sample_starts = build_sample_starts(len(valid_records), args.seq_length)
    train_starts, val_starts, test_starts = split_sample_starts(sample_starts, args.train_ratio, args.val_ratio)
    # Bobot kelas positif agar loss lebih memperhatikan piksel hotspot yang langka.
    pos_weight: float | None = None
    # Proporsi piksel hotspot/risk map positif; nilainya kecil karena class imbalance ekstrem.
    positive_ratio: float | None = None
    if args.with_positive_weight:
        pos_weight, positive_ratio = estimate_positive_weight(
            valid_records,
            train_starts,
            args.seq_length,
            args.native_width,
            args.native_height,
            args.label_dilation_kernel,
            args.label_blur_radius,
            args.max_pos_weight,
        )

    # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
    return {
        "dataset_dir": str(dataset_dir),
        "image_extensions": list(image_extensions),
        "record_count_total": len(records),
        "record_count_valid": len(valid_records),
        "record_count_skipped": len(skipped_records),
        "date_start": valid_records[0].date.date().isoformat(),
        "date_end": valid_records[-1].date.date().isoformat(),
        "seq_length": args.seq_length,
        "sample_count_total": len(sample_starts),
        "train_samples": len(train_starts),
        "val_samples": len(val_starts),
        "test_samples": len(test_starts),
        "positive_mass_ratio_train": positive_ratio,
        "positive_class_weight": pos_weight,
        "positive_weight_computed": args.with_positive_weight,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio_effective": len(test_starts) / max(len(sample_starts), 1),
        "skipped_records_preview": skipped_records[:10],
        "first_record": asdict(valid_records[0]) | {"path": str(valid_records[0].path)},
        "last_record": asdict(valid_records[-1]) | {"path": str(valid_records[-1].path)},
    }


# Fungsi `print_human_summary` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def print_human_summary(summary: dict, show_skipped: bool) -> None:
    print("[data_understanding] Dataset ditemukan:", f"{summary['record_count_valid']} frame valid")
    print("[data_understanding] Total file cocok ekstensi:", summary["record_count_total"])
    print("[data_understanding] Rentang data:", f"{summary['date_start']} s.d. {summary['date_end']}")
    print("[data_understanding] Seq length:", summary["seq_length"])
    print("[data_understanding] Total sample sequence:", summary["sample_count_total"])
    print("[data_understanding] Train samples:", summary["train_samples"])
    print("[data_understanding] Val samples:", summary["val_samples"])
    print("[data_understanding] Test samples:", summary["test_samples"])
    if summary["positive_weight_computed"]:
        print(
            "[data_understanding] Positive mass ratio (train):",
            f"{summary['positive_mass_ratio_train']:.8f}",
        )
        print(
            "[data_understanding] Positive class weight:",
            f"{summary['positive_class_weight']:.4f}",
        )
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        print("[data_understanding] Positive mass ratio (train): dilewati")
        print("[data_understanding] Positive class weight: dilewati")
        print("[data_understanding] Tips: pakai --with-positive-weight untuk menghitung nilai exact.")

    if summary["record_count_skipped"] > 0:
        print("[data_understanding] File dilewati:", summary["record_count_skipped"])
        if show_skipped:
            # Loop ini memproses setiap citra hotspot yang telah terurut kronologis.
            for item in summary["skipped_records_preview"]:
                print(f"- skip {item['path']} | alasan: {item['reason']}")


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> None:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = build_parser()
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parser.parse_args()
    # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
    summary = build_summary(args)
    print_human_summary(summary, show_skipped=args.show_skipped)
    if args.json:
        print()
        # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
