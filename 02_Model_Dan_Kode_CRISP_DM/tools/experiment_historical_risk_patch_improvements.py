
"""Komentar file skripsi:
Tool eksperimen tambahan untuk menganalisis class imbalance, patch lebih besar, dan baseline sederhana.

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
# dataclass dipakai untuk merepresentasikan satu record citra hotspot secara terstruktur.
from dataclasses import dataclass
# datetime dipakai untuk menjaga urutan kronologis citra dan menghitung target H+1.
from datetime import datetime
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path

# NumPy dipakai untuk mask hotspot, tensor sequence, patch, probability map, dan metrik piksel.
import numpy as np
# Pillow Image dipakai untuk membuka citra hotspot PNG/JPG dan menyimpan preview/hasil.
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# Regex ini mengambil tanggal dari nama file dataset FIRMS agar urutan temporal dapat dibentuk.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")
DEFAULT_DATASET_DIR = Path("Ipynb") / "Dataset History Fire Hotspot In Riau Province PNG"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "experiments" / "historical_risk_patch_improvements"


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
    return tuple(extensions) if extensions else (".png", ".jpg", ".jpeg")


# Membaca file citra hotspot dari folder dataset dan mengurutkannya berdasarkan tanggal pada nama file.
def load_records(dataset_dir: Path, image_extensions: tuple[str, ...]) -> list[DatasetRecord]:
    # Daftar record citra hotspot yang sudah terbaca dari folder dataset.
    records: list[DatasetRecord] = []
    allowed = set(image_extensions)
    for path in sorted(item for item in dataset_dir.iterdir() if item.is_file()):
        # File dilewati jika ekstensinya bukan citra yang diizinkan untuk dataset hotspot.
        if path.suffix.lower() not in allowed:
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
    valid: list[DatasetRecord] = []
    skipped: list[dict[str, str]] = []
    # Loop ini memproses setiap citra hotspot yang telah terurut kronologis.
    for record in records:
        try:
            # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
            with Image.open(record.path) as image:
                # Verifikasi ini memastikan file gambar tidak rusak sebelum masuk pipeline.
                image.verify()
            valid.append(record)
        except Exception as exc:
            skipped.append({"path": str(record.path), "reason": str(exc)})
    # Hasil ini dikembalikan sebagai output fungsi `validate_records` untuk tahap berikutnya.
    return valid, skipped


# Membentuk indeks awal sliding window: tujuh citra historis sebagai input dan frame berikutnya sebagai target H+1.
def build_sample_starts(record_count: int, seq_length: int) -> list[int]:
    if record_count <= seq_length:
        # Hasil ini dikembalikan sebagai output fungsi `build_sample_starts` untuk tahap berikutnya.
        return []
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
        raise ValueError("Split train/val/test gagal. Periksa jumlah data atau rasio split.")
    # Hasil ini dikembalikan sebagai output fungsi `split_sample_starts` untuk tahap berikutnya.
    return train, val, test


# Fungsi `normalize_kernel` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def normalize_kernel(size: int) -> int:
    size = max(1, int(size))
    # Hasil ini dikembalikan sebagai output fungsi `normalize_kernel` untuk tahap berikutnya.
    return size if size % 2 == 1 else size + 1


# Fungsi `extract_red_mask` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def extract_red_mask(path: Path, dilation_kernel: int) -> np.ndarray:
    dilation_kernel = normalize_kernel(dilation_kernel)
    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    with Image.open(path) as image:
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
    # Hasil ini dikembalikan sebagai output fungsi `extract_red_mask` untuk tahap berikutnya.
    return np.asarray(mask_image, dtype=np.float32) / 255.0


# Fungsi `build_risk_map` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def build_risk_map(path: Path, dilation_kernel: int, blur_radius: float) -> np.ndarray:
    # `risk` menyimpan peta risiko/target yang dipelajari atau divisualisasikan.
    risk = Image.fromarray(np.uint8(extract_red_mask(path, dilation_kernel=1) * 255.0))
    dilation_kernel = normalize_kernel(dilation_kernel)
    if dilation_kernel > 1:
        # `risk` menyimpan peta risiko/target yang dipelajari atau divisualisasikan.
        risk = risk.filter(ImageFilter.MaxFilter(size=dilation_kernel))
    if blur_radius > 0:
        # `risk` menyimpan peta risiko/target yang dipelajari atau divisualisasikan.
        risk = risk.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    # Hasil ini dikembalikan sebagai output fungsi `build_risk_map` untuk tahap berikutnya.
    return np.asarray(risk, dtype=np.float32) / 255.0


# Fungsi `crop_array` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def crop_array(
    array: np.ndarray,
    cy: int,
    cx: int,
    crop_height: int,
    crop_width: int,
    output_height: int | None = None,
    output_width: int | None = None,
    resample: int = Image.Resampling.BILINEAR,
) -> np.ndarray:
    if array.ndim == 2:
        array = array[..., None]

    height, width = array.shape[:2]
    half_h = crop_height // 2
    half_w = crop_width // 2
    top = cy - half_h
    left = cx - half_w
    bottom = top + crop_height
    right = left + crop_width

    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - height)
    pad_right = max(0, right - width)
    if pad_top or pad_left or pad_bottom or pad_right:
        array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
        top += pad_top
        left += pad_left
        bottom = top + crop_height
        right = left + crop_width

    # Potongan citra/mask berukuran tetap yang masuk ke model atau preview.
    patch = array[top:bottom, left:right, :].astype(np.float32)
    if output_height is None or output_width is None or (output_height == crop_height and output_width == crop_width):
        # Hasil ini dikembalikan sebagai output fungsi `crop_array` untuk tahap berikutnya.
        return patch

    resized_channels: list[np.ndarray] = []
    for channel_idx in range(patch.shape[-1]):
        # Nilai dikunci ke rentang 0-1 agar cocok sebagai mask/probability map.
        image = Image.fromarray(np.uint8(np.clip(patch[:, :, channel_idx], 0.0, 1.0) * 255.0))
        image = image.resize((output_width, output_height), resample)
        resized_channels.append(np.asarray(image, dtype=np.float32) / 255.0)
    # Hasil ini dikembalikan sebagai output fungsi `crop_array` untuk tahap berikutnya.
    return np.stack(resized_channels, axis=-1)


# Memilih pusat patch positif dan negatif agar training melihat hotspot dan background secara seimbang.
def sample_patch_centers(
    target: np.ndarray,
    positive_count: int,
    negative_count: int,
    ground_truth_threshold: float,
    rng: np.random.Generator,
) -> list[tuple[int, int, str]]:
    # Mask boolean yang menandai area target di atas threshold ground truth.
    binary = target >= ground_truth_threshold
    # Koordinat piksel/area yang mengandung target hotspot untuk patch positif.
    positive_coords = np.argwhere(binary)
    # Daftar pusat patch yang akan dipotong dari citra besar.
    centers: list[tuple[int, int, str]] = []

    if len(positive_coords) > 0 and positive_count > 0:
        replace = len(positive_coords) < positive_count
        indices = rng.choice(len(positive_coords), size=positive_count, replace=replace)
        for idx in np.atleast_1d(indices):
            cy, cx = positive_coords[int(idx)]
            centers.append((int(cy), int(cx), "positive"))

    attempts = max(negative_count * 60, 60)
    while negative_count > 0 and attempts > 0:
        cy = int(rng.integers(0, target.shape[0]))
        cx = int(rng.integers(0, target.shape[1]))
        if not binary[cy, cx]:
            centers.append((cy, cx, "negative"))
            negative_count -= 1
        attempts -= 1

    # Hasil ini dikembalikan sebagai output fungsi `sample_patch_centers` untuk tahap berikutnya.
    return centers


# Fungsi `metrics_from_counts` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "accuracy": accuracy,
    }


# Fungsi `count_binary_metrics` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def count_binary_metrics(
    truth_map: np.ndarray,
    pred_map: np.ndarray,
    threshold: float,
    ground_truth_threshold: float,
) -> tuple[int, int, int, int]:
    truth = truth_map >= ground_truth_threshold
    pred = pred_map >= threshold
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())
    # Hasil ini dikembalikan sebagai output fungsi `count_binary_metrics` untuk tahap berikutnya.
    return tp, fp, fn, tn


# Fungsi `dilate_bool` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        # Hasil ini dikembalikan sebagai output fungsi `dilate_bool` untuk tahap berikutnya.
        return mask.astype(bool)
    kernel = normalize_kernel((int(radius) * 2) + 1)
    image = Image.fromarray((mask.astype(bool).astype(np.uint8)) * 255)
    # MaxFilter memperluas area positif sehingga hotspot kecil tidak hilang saat dipatch.
    image = image.filter(ImageFilter.MaxFilter(size=kernel))
    # Hasil ini dikembalikan sebagai output fungsi `dilate_bool` untuk tahap berikutnya.
    return np.asarray(image, dtype=np.uint8) > 0


# Fungsi `count_buffered_metrics` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def count_buffered_metrics(
    truth_map: np.ndarray,
    pred_map: np.ndarray,
    threshold: float,
    ground_truth_threshold: float,
    buffer_radius: int,
) -> tuple[int, int, int, int]:
    truth = dilate_bool(truth_map >= ground_truth_threshold, buffer_radius)
    pred = dilate_bool(pred_map >= threshold, buffer_radius)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())
    # Hasil ini dikembalikan sebagai output fungsi `count_buffered_metrics` untuk tahap berikutnya.
    return tp, fp, fn, tn


# Fungsi `threshold_values` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def threshold_values(step: float) -> list[float]:
    if step <= 0 or step >= 1:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("threshold-step harus berada pada rentang 0 sampai 1.")
    # Hasil ini dikembalikan sebagai output fungsi `threshold_values` untuk tahap berikutnya.
    return [round(float(value), 4) for value in np.arange(step, 1.0, step)]


# Fungsi `metric_rank` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def metric_rank(metrics: dict[str, float | int]) -> tuple[float, float, float, float, int]:
    # Hasil ini dikembalikan sebagai output fungsi `metric_rank` untuk tahap berikutnya.
    return (
        float(metrics["f1_score"]),
        float(metrics["iou"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -int(metrics["fp"]),
    )


# Fungsi `prepare_records` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def prepare_records(args: argparse.Namespace) -> tuple[list[DatasetRecord], list[int], list[int], list[int]]:
    # `dataset_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    records, skipped = validate_records(load_records(dataset_dir, parse_image_extensions(args.image_extensions)))
    if skipped:
        print(f"[warning] {len(skipped)} file dilewati karena tidak valid.")
    # Pengecekan ini memastikan jumlah citra cukup untuk membentuk sequence tujuh hari dan target H+1.
    if len(records) <= args.seq_length:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Jumlah record tidak cukup untuk membentuk sequence.")
    # Indeks awal tiap sequence input H-6 sampai H0 untuk memprediksi target H+1.
    sample_starts = build_sample_starts(len(records), args.seq_length)
    train, val, test = split_sample_starts(sample_starts, args.train_ratio, args.val_ratio)
    # Hasil ini dikembalikan sebagai output fungsi `prepare_records` untuk tahap berikutnya.
    return records, train, val, test


# Fungsi `select_split` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def select_split(args: argparse.Namespace, train: list[int], val: list[int], test: list[int]) -> list[int]:
    split_map = {"train": train, "val": val, "test": test}
    selected = split_map[args.split]
    if args.max_samples and args.max_samples > 0:
        # Hasil ini dikembalikan sebagai output fungsi `select_split` untuk tahap berikutnya.
        return selected[: args.max_samples]
    # Hasil ini dikembalikan sebagai output fungsi `select_split` untuk tahap berikutnya.
    return selected


# Fungsi `run_imbalance` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def run_imbalance(args: argparse.Namespace) -> dict:
    records, train, val, test = prepare_records(args)
    selected = select_split(args, train, val, test)
    rng = np.random.default_rng(args.seed)
    # `patch_sizes` berkaitan dengan potongan citra 160x160 yang dipakai model.
    patch_sizes = [int(item.strip()) for item in args.patch_sizes.split(",") if item.strip()]

    positive_pixels = 0
    total_pixels = 0
    # `patch_stats` berkaitan dengan potongan citra 160x160 yang dipakai model.
    patch_stats = {
        size: {"positive_patch_ratios": [], "negative_patch_ratios": []}
        for size in patch_sizes
    }

    for start in selected:
        target_record = records[start + args.seq_length]
        target = build_risk_map(target_record.path, args.label_dilation_kernel, args.label_blur_radius)
        # Mask boolean yang menandai area target di atas threshold ground truth.
        binary = target >= args.ground_truth_threshold
        positive_pixels += int(binary.sum())
        total_pixels += int(binary.size)

        # Daftar pusat patch yang akan dipotong dari citra besar.
        centers = sample_patch_centers(
            target=target,
            positive_count=args.train_positive_patches,
            negative_count=args.train_negative_patches,
            # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
            ground_truth_threshold=args.ground_truth_threshold,
            rng=rng,
        )
        for size in patch_sizes:
            for cy, cx, kind in centers:
                # Potongan citra/mask berukuran tetap yang masuk ke model atau preview.
                patch = crop_array(target, cy, cx, size, size)[:, :, 0]
                ratio = float((patch >= args.ground_truth_threshold).mean())
                if kind == "positive":
                    patch_stats[size]["positive_patch_ratios"].append(ratio)
                # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
                else:
                    patch_stats[size]["negative_patch_ratios"].append(ratio)

    negative_pixels = total_pixels - positive_pixels
    # Proporsi piksel hotspot/risk map positif; nilainya kecil karena class imbalance ekstrem.
    positive_ratio = 0.0 if total_pixels == 0 else positive_pixels / total_pixels
    raw_pos_weight = 0.0 if positive_pixels == 0 else negative_pixels / positive_pixels
    capped_pos_weight = min(raw_pos_weight, args.max_pos_weight)

    # `patch_summary` berkaitan dengan potongan citra 160x160 yang dipakai model.
    patch_summary = {}
    for size, values in patch_stats.items():
        pos_values = values["positive_patch_ratios"]
        neg_values = values["negative_patch_ratios"]
        # `patch_summary[str(size)]` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_summary[str(size)] = {
            "positive_patch_count": len(pos_values),
            "negative_patch_count": len(neg_values),
            "mean_positive_patch_positive_ratio": float(np.mean(pos_values)) if pos_values else 0.0,
            "median_positive_patch_positive_ratio": float(np.median(pos_values)) if pos_values else 0.0,
            "mean_negative_patch_positive_ratio": float(np.mean(neg_values)) if neg_values else 0.0,
        }

    report = {
        "mode": "imbalance",
        "split": args.split,
        "sample_count": len(selected),
        "dataset_frame_count": len(records),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "positive_pixels": positive_pixels,
        "negative_pixels": negative_pixels,
        "total_pixels": total_pixels,
        "positive_ratio": positive_ratio,
        "raw_positive_class_weight": raw_pos_weight,
        "capped_positive_class_weight": capped_pos_weight,
        "recommended_framing": "prediksi area risiko hotspot, bukan prediksi titik hotspot presisi tinggi",
        "patch_summary": patch_summary,
        "practical_recommendation": {
            "increase_positive_patch_sampling": "coba 6:1 atau 8:1 untuk train-positive-patches:train-negative-patches",
            "larger_patch_candidates": patch_sizes,
            "use_feature_stack": "mask_context3 untuk konteks tambahan yang sudah tersedia di script training utama",
        },
    }
    write_report(report, args.output_dir, "imbalance_patch_report.json")
    print_json_or_pretty(report, args.json)
    # Hasil ini dikembalikan sebagai output fungsi `run_imbalance` untuk tahap berikutnya.
    return report


# Fungsi `run_baseline` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def run_baseline(args: argparse.Namespace) -> dict:
    records, train, val, test = prepare_records(args)
    selected = select_split(args, train, val, test)
    # `thresholds` berkaitan dengan ambang untuk membedakan area risiko dan background.
    thresholds = threshold_values(args.threshold_step)
    counts = {threshold: [0, 0, 0, 0] for threshold in thresholds}

    for start in selected:
        h0_record = records[start + args.seq_length - 1]
        target_record = records[start + args.seq_length]
        pred = build_risk_map(h0_record.path, args.baseline_dilation_kernel, args.baseline_blur_radius)
        truth = build_risk_map(target_record.path, args.label_dilation_kernel, args.label_blur_radius)
        # Loop ini mencoba beberapa threshold untuk mencari ambang evaluasi terbaik.
        for threshold in thresholds:
            tp, fp, fn, tn = count_binary_metrics(truth, pred, threshold, args.ground_truth_threshold)
            bucket = counts[threshold]
            bucket[0] += tp
            bucket[1] += fp
            bucket[2] += fn
            bucket[3] += tn

    sweep = []
    # `best_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
    best_threshold = thresholds[0]
    # `best_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    best_metrics: dict[str, float | int] | None = None
    # Loop ini mencoba beberapa threshold untuk mencari ambang evaluasi terbaik.
    for threshold in thresholds:
        tp, fp, fn, tn = counts[threshold]
        # Nilai precision, recall, F1, IoU, dan metrik lain hasil evaluasi.
        metrics = metrics_from_counts(tp, fp, fn, tn)
        row = {"threshold": threshold, **metrics}
        sweep.append(row)
        if best_metrics is None or metric_rank(metrics) > metric_rank(best_metrics):
            # `best_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
            best_threshold = threshold
            # `best_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
            best_metrics = metrics

    assert best_metrics is not None

    buffered_counts = [0, 0, 0, 0]
    for start in selected:
        h0_record = records[start + args.seq_length - 1]
        target_record = records[start + args.seq_length]
        pred = build_risk_map(h0_record.path, args.baseline_dilation_kernel, args.baseline_blur_radius)
        truth = build_risk_map(target_record.path, args.label_dilation_kernel, args.label_blur_radius)
        tp, fp, fn, tn = count_buffered_metrics(
            truth,
            pred,
            best_threshold,
            args.ground_truth_threshold,
            args.buffer_radius,
        )
        buffered_counts[0] += tp
        buffered_counts[1] += fp
        buffered_counts[2] += fn
        buffered_counts[3] += tn

    # `buffered_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    buffered_metrics = {
        "buffer_radius": args.buffer_radius,
        **metrics_from_counts(*buffered_counts),
    }
    report = {
        "mode": "persistence_baseline",
        "baseline_definition": "prediksi H+1 = risk map dari H0 yang diberi dilation/blur",
        "split": args.split,
        "sample_count": len(selected),
        "best_threshold": best_threshold,
        "best_standard_metrics": best_metrics,
        "best_buffered_metrics": buffered_metrics,
        "sweep": sweep,
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "interpretation": (
            "Jika ConvLSTM tidak jauh lebih baik dari baseline ini, maka model belum menangkap "
            "pola temporal yang kuat di luar persistensi hotspot H0."
        ),
    }
    write_report(report, args.output_dir, "h0_persistence_baseline_report.json")
    print_json_or_pretty(report, args.json)
    # Hasil ini dikembalikan sebagai output fungsi `run_baseline` untuk tahap berikutnya.
    return report


# Fungsi `find_record_with_hotspot` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def find_record_with_hotspot(records: list[DatasetRecord], dilation_kernel: int) -> DatasetRecord:
    best_record: DatasetRecord | None = None
    best_count = -1
    # Loop ini memproses setiap citra hotspot yang telah terurut kronologis.
    for record in records:
        # Mask biner hotspot: piksel merah menjadi positif dan background menjadi nol.
        mask = extract_red_mask(record.path, dilation_kernel)
        count = int((mask > 0).sum())
        if count > best_count:
            best_record = record
            best_count = count
    if best_record is None or best_count <= 0:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Tidak ditemukan citra dengan hotspot merah.")
    # Hasil ini dikembalikan sebagai output fungsi `find_record_with_hotspot` untuk tahap berikutnya.
    return best_record


# Fungsi `draw_centered_text` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    draw.text(
        (x0 + (x1 - x0 - text_width) / 2, y0 + (y1 - y0 - text_height) / 2),
        text,
        fill=fill,
        font=font,
    )


# Fungsi `save_multiscale_preview` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def save_multiscale_preview(args: argparse.Namespace) -> dict:
    records, train, val, test = prepare_records(args)
    source = Path(args.source_image).expanduser().resolve() if args.source_image else None
    if source and source.exists():
        record = DatasetRecord(path=source, date=datetime.min)
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        record = find_record_with_hotspot(records, args.input_dilation_kernel)

    # Mask biner hotspot: piksel merah menjadi positif dan background menjadi nol.
    mask = extract_red_mask(record.path, args.input_dilation_kernel)
    # Koordinat piksel/area yang mengandung target hotspot untuk patch positif.
    positive_coords = np.argwhere(mask >= args.ground_truth_threshold)
    if len(positive_coords) == 0:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Citra terpilih tidak punya piksel hotspot setelah threshold.")
    center_idx = int(len(positive_coords) // 2)
    cy, cx = positive_coords[center_idx]
    cy = int(cy)
    cx = int(cx)

    # `patch_size` berkaitan dengan potongan citra 160x160 yang dipakai model.
    patch_size = int(args.patch_size)
    context_size = int(round(patch_size * float(args.context_scale)))
    # `local_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
    local_mask = crop_array(mask, cy, cx, patch_size, patch_size, patch_size, patch_size, Image.Resampling.NEAREST)
    # `context_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
    context_mask = crop_array(mask, cy, cx, context_size, context_size, patch_size, patch_size, Image.Resampling.BILINEAR)
    multiscale = np.concatenate([local_mask, context_mask], axis=-1)

    # `output_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # `local_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    local_path = output_dir / f"multiscale_local_mask_{patch_size}.png"
    # `context_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    context_path = output_dir / f"multiscale_context_mask_{patch_size}_scale_{args.context_scale}.png"
    # `stacked_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    stacked_path = output_dir / f"multiscale_two_channel_preview_{patch_size}.png"
    # `composite_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    composite_path = output_dir / f"multiscale_patch_preview_{patch_size}.png"

    Image.fromarray(np.uint8(local_mask[:, :, 0] * 255)).save(local_path)
    Image.fromarray(np.uint8(context_mask[:, :, 0] * 255)).save(context_path)

    stacked_rgb = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    stacked_rgb[:, :, 0] = np.uint8(local_mask[:, :, 0] * 255)
    stacked_rgb[:, :, 1] = np.uint8(context_mask[:, :, 0] * 255)
    Image.fromarray(stacked_rgb).save(stacked_path)

    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    with Image.open(record.path) as image:
        source_rgb = image.convert("RGB")
    preview = source_rgb.copy()
    draw = ImageDraw.Draw(preview, "RGBA")

    # Fungsi `box` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def box(size: int) -> tuple[int, int, int, int]:
        left = max(0, cx - size // 2)
        top = max(0, cy - size // 2)
        right = min(preview.width, left + size)
        bottom = min(preview.height, top + size)
        # Hasil ini dikembalikan sebagai output fungsi `box` untuk tahap berikutnya.
        return left, top, right, bottom

    draw.rectangle(box(context_size), outline=(255, 220, 0, 255), width=6)
    draw.rectangle(box(patch_size), outline=(255, 0, 0, 255), width=6)
    preview.thumbnail((920, 460), Image.Resampling.LANCZOS)

    panel_w = patch_size
    header_h = 34
    composite = Image.new("RGB", (max(920, panel_w * 3), preview.height + panel_w + header_h * 2 + 36), (242, 246, 250))
    d = ImageDraw.Draw(composite)
    composite.paste(preview, (0, 0))
    y = preview.height + 18
    labels = ["Local detail mask", "Large context mask", "2-channel preview"]
    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    images = [Image.open(local_path).convert("RGB"), Image.open(context_path).convert("RGB"), Image.open(stacked_path).convert("RGB")]
    for idx, (label, item) in enumerate(zip(labels, images)):
        x = idx * panel_w
        d.rectangle((x, y, x + panel_w, y + header_h), fill=(30, 68, 102))
        draw_centered_text(d, (x, y, x + panel_w, y + header_h), label, (255, 255, 255))
        composite.paste(item, (x, y + header_h))
    composite.save(composite_path)

    report = {
        "mode": "preview_multiscale",
        "source_image": str(record.path),
        "patch_size": patch_size,
        "context_size": context_size,
        "context_scale": args.context_scale,
        "multiscale_tensor_shape": list(multiscale.shape),
        "meaning": {
            "channel_0": "mask hotspot lokal untuk detail",
            "channel_1": "mask hotspot konteks lebih luas yang di-resize ke ukuran patch",
        },
        "outputs": {
            "local_mask": str(local_path),
            "context_mask": str(context_path),
            "two_channel_preview": str(stacked_path),
            "composite": str(composite_path),
        },
    }
    write_report(report, args.output_dir, "multiscale_preview_report.json")
    print_json_or_pretty(report, args.json)
    # Hasil ini dikembalikan sebagai output fungsi `box` untuk tahap berikutnya.
    return report


# Fungsi `run_commands` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def run_commands(args: argparse.Namespace) -> dict:
    # `dataset_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    dataset_dir = args.dataset_dir
    commands = {
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "baseline": (
            'python -u tools/experiment_historical_risk_patch_improvements.py baseline '
            f'--dataset-dir "{dataset_dir}" --split test --threshold-step 0.05'
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "imbalance_patch_224_preview": (
            'python -u tools/experiment_historical_risk_patch_improvements.py imbalance '
            f'--dataset-dir "{dataset_dir}" --patch-sizes 160,224,256 '
            "--train-positive-patches 6 --train-negative-patches 1 --split train --max-samples 120"
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "multiscale_preview": (
            'python -u tools/experiment_historical_risk_patch_improvements.py preview-multiscale '
            f'--dataset-dir "{dataset_dir}" --patch-size 224 --context-scale 2.0'
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "training_patch_224_wsl": (
            "bash run_train_historical_risk_patch_wsl_gpu.sh "
            f'--dataset-dir "$HOME/projects/sistem-web-skripsi-ta/Ipynb/Dataset History Fire Hotspot In Riau Province PNG" '
            "--image-extensions .png --patch-width 224 --patch-height 224 --batch-size 1 --epochs 3 "
            "--train-positive-patches 6 --train-negative-patches 1 "
            "--input-dilation-kernel 5 --label-dilation-kernel 9 --label-blur-radius 2.0 "
            "--evaluation-buffer-radius 5 --threshold 0.55 --ground-truth-threshold 0.05 "
            "--loss-strategy wbce_dice_context --feature-stack mask_context3 --context-kernel 15 "
            "--enable-augmentation --augmentation-rotate90"
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "training_patch_256_wsl": (
            "bash run_train_historical_risk_patch_wsl_gpu.sh "
            f'--dataset-dir "$HOME/projects/sistem-web-skripsi-ta/Ipynb/Dataset History Fire Hotspot In Riau Province PNG" '
            "--image-extensions .png --patch-width 256 --patch-height 256 --batch-size 1 --epochs 3 "
            "--train-positive-patches 6 --train-negative-patches 1 "
            "--input-dilation-kernel 5 --label-dilation-kernel 9 --label-blur-radius 2.0 "
            "--evaluation-buffer-radius 5 --threshold 0.55 --ground-truth-threshold 0.05 "
            "--loss-strategy wbce_dice_context --feature-stack mask_context3 --context-kernel 15 "
            "--enable-augmentation --augmentation-rotate90"
        ),
    }
    print_json_or_pretty(commands, args.json)
    # Hasil ini dikembalikan sebagai output fungsi `run_commands` untuk tahap berikutnya.
    return commands


# Fungsi `write_report` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def write_report(report: dict, output_dir: str | Path, file_name: str) -> None:
    # `output_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    (output_path / file_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# Fungsi `print_json_or_pretty` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def print_json_or_pretty(data: dict, as_json: bool) -> None:
    if as_json:
        # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
        print(json.dumps(data, indent=2, ensure_ascii=False))
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    mode = data.get("mode", "commands")
    print(f"[experiment] Mode: {mode}")
    if mode == "imbalance":
        print(f"[experiment] Split: {data['split']} | samples: {data['sample_count']}")
        print(f"[experiment] Positive ratio: {data['positive_ratio']:.8f}")
        print(f"[experiment] Raw positive class weight: {data['raw_positive_class_weight']:.4f}")
        print(f"[experiment] Capped positive class weight: {data['capped_positive_class_weight']:.4f}")
        print("[experiment] Patch summary:")
        for size, row in data["patch_summary"].items():
            print(
                "  - "
                f"{size}x{size}: pos_patch_mean={row['mean_positive_patch_positive_ratio']:.6f}, "
                # `f"neg_patch_mean` berkaitan dengan potongan citra 160x160 yang dipakai model.
                f"neg_patch_mean={row['mean_negative_patch_positive_ratio']:.6f}"
            )
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return
    if mode == "persistence_baseline":
        standard = data["best_standard_metrics"]
        buffered = data["best_buffered_metrics"]
        print(f"[experiment] Split: {data['split']} | samples: {data['sample_count']}")
        print(f"[experiment] Best threshold: {data['best_threshold']}")
        print(
            "[experiment] Standard: "
            f"precision={standard['precision']:.4f}, recall={standard['recall']:.4f}, "
            f"f1={standard['f1_score']:.4f}, iou={standard['iou']:.4f}"
        )
        print(
            "[experiment] Buffered: "
            f"precision={buffered['precision']:.4f}, recall={buffered['recall']:.4f}, "
            f"f1={buffered['f1_score']:.4f}, iou={buffered['iou']:.4f}"
        )
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return
    if mode == "preview_multiscale":
        print(f"[experiment] Source: {data['source_image']}")
        print(f"[experiment] Tensor shape: {data['multiscale_tensor_shape']}")
        for key, path in data["outputs"].items():
            print(f"[experiment] {key}: {path}")
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    for key, value in data.items():
        print(f"[{key}]")
        print(value)


# Fungsi `add_common_args` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def add_common_args(parser: argparse.ArgumentParser) -> None:
    # Opsi ini menentukan folder citra hotspot historis yang akan diproses.
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    # Opsi ini membatasi ekstensi citra yang dibaca dari dataset.
    parser.add_argument("--image-extensions", default=".png")
    # Opsi ini menentukan jumlah frame historis; project memakai 7 frame H-6 sampai H0.
    parser.add_argument("--seq-length", type=int, default=7)
    # Opsi ini mengatur porsi sequence untuk training secara kronologis.
    parser.add_argument("--train-ratio", type=float, default=0.70)
    # Opsi ini mengatur porsi sequence untuk validation setelah training.
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # Opsi `--split` menambah parameter eksekusi script.
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    # Opsi `--max-samples` menambah parameter eksekusi script.
    parser.add_argument("--max-samples", type=int, default=0)
    # Opsi ini menentukan batas target risk map yang dihitung sebagai positif.
    parser.add_argument("--ground-truth-threshold", type=float, default=0.05)
    # Opsi ini memperluas label hotspot agar target H+1 tidak terlalu kecil.
    parser.add_argument("--label-dilation-kernel", type=int, default=9)
    # Opsi ini menghaluskan target risk map supaya model belajar area risiko, bukan titik kaku.
    parser.add_argument("--label-blur-radius", type=float, default=2.0)
    # Opsi ini mempertebal mask input agar hotspot kecil tidak hilang.
    parser.add_argument("--input-dilation-kernel", type=int, default=5)
    # Opsi ini menentukan folder output preview, laporan, atau hasil validasi.
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    # Opsi ini membuat output ringkasan dicetak sebagai JSON agar mudah dikutip/dibaca ulang.
    parser.add_argument("--json", action="store_true")


# Menyusun opsi command line agar fase CRISP-DM bisa dijalankan ulang dengan parameter yang jelas.
def build_parser() -> argparse.ArgumentParser:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description=(
            "Eksperimen terpisah untuk peningkatan historical_risk_patch: "
            "class imbalance, patch besar, multi-scale preview, baseline H0, dan evaluasi area risiko."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    imbalance = subparsers.add_parser("imbalance", help="Ringkas class imbalance dan efek kandidat patch besar.")
    add_common_args(imbalance)
    imbalance.add_argument("--patch-sizes", default="160,224,256")
    imbalance.add_argument("--train-positive-patches", type=int, default=6)
    imbalance.add_argument("--train-negative-patches", type=int, default=1)
    imbalance.add_argument("--max-pos-weight", type=float, default=50.0)
    imbalance.add_argument("--seed", type=int, default=42)
    imbalance.set_defaults(func=run_imbalance, split="train")

    baseline = subparsers.add_parser("baseline", help="Evaluasi baseline sederhana: prediksi H+1 = H0 dilated.")
    add_common_args(baseline)
    baseline.add_argument("--baseline-dilation-kernel", type=int, default=9)
    baseline.add_argument("--baseline-blur-radius", type=float, default=2.0)
    baseline.add_argument("--threshold-step", type=float, default=0.05)
    baseline.add_argument("--buffer-radius", type=int, default=5)
    baseline.set_defaults(func=run_baseline)

    preview = subparsers.add_parser("preview-multiscale", help="Buat visualisasi patch lokal + konteks lebih luas.")
    add_common_args(preview)
    preview.add_argument("--patch-size", type=int, default=224)
    preview.add_argument("--context-scale", type=float, default=2.0)
    preview.add_argument("--source-image", default="")
    preview.set_defaults(func=save_multiscale_preview)

    commands = subparsers.add_parser("commands", help="Cetak command eksperimen yang disarankan.")
    commands.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    commands.add_argument("--json", action="store_true")
    commands.set_defaults(func=run_commands)
    # Hasil ini dikembalikan sebagai output fungsi `build_parser` untuk tahap berikutnya.
    return parser


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> None:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = build_parser()
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
