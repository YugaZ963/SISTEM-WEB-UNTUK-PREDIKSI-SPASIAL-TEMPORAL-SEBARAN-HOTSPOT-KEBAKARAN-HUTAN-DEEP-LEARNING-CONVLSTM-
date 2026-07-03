
"""Komentar file skripsi:
Tool evaluasi metrik dasar ConvLSTM untuk precision, recall, F1-score, IoU, dan accuracy.

Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
terutama data understanding, data preparation, modeling, evaluation,
atau deployment sesuai fungsi teknisnya.
"""

from __future__ import annotations

# argparse dipakai agar parameter dataset, patch, threshold, dan output bisa diatur dari command line.
import argparse
# json dipakai untuk menyimpan ringkasan eksperimen, metrik, dan metadata proses.
import json
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path

# NumPy dipakai untuk mask hotspot, tensor sequence, patch, probability map, dan metrik piksel.
import numpy as np
# Pillow Image dipakai untuk membuka citra hotspot PNG/JPG dan menyimpan preview/hasil.
from PIL import Image


# Membaca semua parameter training seperti dataset, patch, threshold, batch, epoch, dan lokasi output.
def parse_args() -> argparse.Namespace:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description="Evaluasi mask prediksi vs ground truth (Precision, Recall, F1, IoU, Accuracy)."
    )
    # Opsi `--pred-dir` menambah parameter eksekusi script.
    parser.add_argument("--pred-dir", required=True, help="Folder mask prediksi.")
    # Opsi `--gt-dir` menambah parameter eksekusi script.
    parser.add_argument("--gt-dir", required=True, help="Folder mask ground truth.")
    # Opsi ini menentukan ambang binary mask pada evaluasi atau visualisasi.
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold biner 0..1.")
    # Opsi `--suffix` menambah parameter eksekusi script.
    parser.add_argument("--suffix", default=".png", help="Ekstensi file mask yang dievaluasi.")
    # Opsi `--output-json` menambah parameter eksekusi script.
    parser.add_argument("--output-json", default="", help="Path output JSON hasil evaluasi.")
    # Hasil ini dikembalikan sebagai output fungsi `parse_args` untuk tahap berikutnya.
    return parser.parse_args()


# Fungsi `load_mask` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def load_mask(path: Path) -> np.ndarray:
    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # Hasil ini dikembalikan sebagai output fungsi `load_mask` untuk tahap berikutnya.
    return arr


# Fungsi `safe_div` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def safe_div(a: float, b: float) -> float:
    if b == 0:
        # Hasil ini dikembalikan sebagai output fungsi `safe_div` untuk tahap berikutnya.
        return 0.0
    # Hasil ini dikembalikan sebagai output fungsi `safe_div` untuk tahap berikutnya.
    return a / b


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> int:
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parse_args()
    # `pred_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    pred_dir = Path(args.pred_dir).resolve()
    # `gt_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    gt_dir = Path(args.gt_dir).resolve()

    if not pred_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError("Folder pred-dir atau gt-dir tidak ditemukan.")

    # Ambang untuk mengubah probability map menjadi binary mask area risiko.
    threshold = float(args.threshold)
    if not (0.0 <= threshold <= 1.0):
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Threshold harus di rentang 0..1")

    pred_files = {p.name: p for p in pred_dir.glob(f"*{args.suffix}") if p.is_file()}
    gt_files = {p.name: p for p in gt_dir.glob(f"*{args.suffix}") if p.is_file()}
    common_names = sorted(set(pred_files).intersection(gt_files))

    if not common_names:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Tidak ada pasangan file prediksi-groundtruth yang cocok.")

    tp = fp = fn = tn = 0
    for name in common_names:
        pred = load_mask(pred_files[name]) >= threshold
        gt = load_mask(gt_files[name]) >= threshold

        if pred.shape != gt.shape:
            # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
            raise ValueError(f"Shape tidak sama untuk file {name}: {pred.shape} vs {gt.shape}")

        tp += int(np.logical_and(pred, gt).sum())
        fp += int(np.logical_and(pred, np.logical_not(gt)).sum())
        fn += int(np.logical_and(np.logical_not(pred), gt).sum())
        tn += int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    # `precision` menyimpan ukuran evaluasi performa prediksi hotspot.
    precision = safe_div(tp, tp + fp)
    # `recall` menyimpan ukuran evaluasi performa prediksi hotspot.
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    # `iou` menyimpan ukuran evaluasi performa prediksi hotspot.
    iou = safe_div(tp, tp + fp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    # Nilai precision, recall, F1, IoU, dan metrik lain hasil evaluasi.
    metrics = {
        "num_files": len(common_names),
        "threshold": threshold,
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

    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    print(json.dumps(metrics, indent=2))

    if args.output_json:
        # `output_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Hasil evaluasi disimpan ke: {output_path}")

    # Hasil ini dikembalikan sebagai output fungsi `main` untuk tahap berikutnya.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
