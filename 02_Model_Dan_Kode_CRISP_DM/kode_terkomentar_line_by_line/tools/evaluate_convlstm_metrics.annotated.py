# File anotasi dari `tools/evaluate_convlstm_metrics.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Tool evaluasi metrik dasar ConvLSTM untuk precision, recall, F1-score, IoU, dan accuracy.

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
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image


# Membuat langkah kerja bernama `parse_args`.
def parse_args() -> argparse.Namespace:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description="Evaluasi mask prediksi vs ground truth (Precision, Recall, F1, IoU, Accuracy)."
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--pred-dir", required=True, help="Folder mask prediksi.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--gt-dir", required=True, help="Folder mask ground truth.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold biner 0..1.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--suffix", default=".png", help="Ekstensi file mask yang dievaluasi.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--output-json", default="", help="Path output JSON hasil evaluasi.")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser.parse_args()


# Membuat langkah kerja bernama `load_mask`.
def load_mask(path: Path) -> np.ndarray:
    # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
    image = Image.open(path).convert("L")
    # Menyimpan nilai ke `arr` untuk dipakai pada langkah berikutnya.
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return arr


# Membuat langkah kerja bernama `safe_div`.
def safe_div(a: float, b: float) -> float:
    # Mengecek syarat sebelum melanjutkan proses.
    if b == 0:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return 0.0
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return a / b


# Membuat langkah kerja bernama `main`.
def main() -> int:
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parse_args()
    # Menyimpan nilai ke `pred_dir` untuk dipakai pada langkah berikutnya.
    pred_dir = Path(args.pred_dir).resolve()
    # Menyimpan nilai ke `gt_dir` untuk dipakai pada langkah berikutnya.
    gt_dir = Path(args.gt_dir).resolve()

    # Mengecek syarat sebelum melanjutkan proses.
    if not pred_dir.exists() or not gt_dir.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError("Folder pred-dir atau gt-dir tidak ditemukan.")

    # Menyimpan nilai ke `threshold` untuk dipakai pada langkah berikutnya.
    threshold = float(args.threshold)
    # Mengecek syarat sebelum melanjutkan proses.
    if not (0.0 <= threshold <= 1.0):
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Threshold harus di rentang 0..1")

    # Menyimpan nilai ke `pred_files` untuk dipakai pada langkah berikutnya.
    pred_files = {p.name: p for p in pred_dir.glob(f"*{args.suffix}") if p.is_file()}
    # Menyimpan nilai ke `gt_files` untuk dipakai pada langkah berikutnya.
    gt_files = {p.name: p for p in gt_dir.glob(f"*{args.suffix}") if p.is_file()}
    # Menyimpan nilai ke `common_names` untuk dipakai pada langkah berikutnya.
    common_names = sorted(set(pred_files).intersection(gt_files))

    # Mengecek syarat sebelum melanjutkan proses.
    if not common_names:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Tidak ada pasangan file prediksi-groundtruth yang cocok.")

    # Menyimpan nilai ke `tp` untuk dipakai pada langkah berikutnya.
    tp = fp = fn = tn = 0
    # Mengulang proses untuk setiap data dalam daftar.
    for name in common_names:
        # Menyimpan nilai ke `pred` untuk dipakai pada langkah berikutnya.
        pred = load_mask(pred_files[name]) >= threshold
        # Menyimpan nilai ke `gt` untuk dipakai pada langkah berikutnya.
        gt = load_mask(gt_files[name]) >= threshold

        # Mengecek syarat sebelum melanjutkan proses.
        if pred.shape != gt.shape:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise ValueError(f"Shape tidak sama untuk file {name}: {pred.shape} vs {gt.shape}")

        # Menyimpan nilai ke `tp +` untuk dipakai pada langkah berikutnya.
        tp += int(np.logical_and(pred, gt).sum())
        # Menyimpan nilai ke `fp +` untuk dipakai pada langkah berikutnya.
        fp += int(np.logical_and(pred, np.logical_not(gt)).sum())
        # Menyimpan nilai ke `fn +` untuk dipakai pada langkah berikutnya.
        fn += int(np.logical_and(np.logical_not(pred), gt).sum())
        # Menyimpan nilai ke `tn +` untuk dipakai pada langkah berikutnya.
        tn += int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

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

    # Menyimpan nilai ke `metrics` untuk dipakai pada langkah berikutnya.
    metrics = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "num_files": len(common_names),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "threshold": threshold,
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

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(json.dumps(metrics, indent=2))

    # Mengecek syarat sebelum melanjutkan proses.
    if args.output_json:
        # Menyimpan nilai ke `output_path` untuk dipakai pada langkah berikutnya.
        output_path = Path(args.output_json).resolve()
        # Menyimpan nilai ke `output_path.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Menyimpan nilai ke `output_path.write_text(json.dumps(metrics, indent` untuk dipakai pada langkah berikutnya.
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"Hasil evaluasi disimpan ke: {output_path}")

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return 0


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Menghentikan proses dan memberi pesan kesalahan yang jelas.
    raise SystemExit(main())
