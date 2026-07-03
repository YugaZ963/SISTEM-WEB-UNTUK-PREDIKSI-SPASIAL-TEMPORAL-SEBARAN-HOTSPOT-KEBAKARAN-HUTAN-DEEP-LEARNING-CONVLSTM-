# File anotasi dari `tools/evaluation_historical_risk_patch.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Evaluation, yaitu mengukur hasil prediksi dan validasi.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Script ringkas fase Evaluation CRISP-DM untuk membaca report training dan menampilkan metrik model.

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
from pathlib import Path

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image, ImageFilter


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")


# Membuat langkah kerja bernama `_normalize_kernel`.
def _normalize_kernel(size: int) -> int:
    # Menyimpan nilai ke `size` untuk dipakai pada langkah berikutnya.
    size = max(1, int(size))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return size if size % 2 == 1 else size + 1


# Membuat langkah kerja bernama `load_native_risk_map`.
def load_native_risk_map(
    # Menjelaskan data `path` yang disimpan atau dikirim pada bagian ini.
    path: Path,
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
    # Mengecek syarat sebelum melanjutkan proses.
    if mask.ndim == 4 and mask.shape[-1] == 1:
        # Menyimpan nilai ke `squeezed` untuk dipakai pada langkah berikutnya.
        squeezed = np.squeeze(mask.astype(np.uint8), axis=-1)
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menyimpan nilai ke `squeezed` untuk dipakai pada langkah berikutnya.
        squeezed = mask.astype(np.uint8)

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
        raise ValueError(f"Shape mask tidak didukung untuk buffered evaluation: {mask.shape}")

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


# Membuat langkah kerja bernama `sweep_thresholds`.
def sweep_thresholds(
    # Menjelaskan data `y_true` yang disimpan atau dikirim pada bagian ini.
    y_true: np.ndarray,
    # Menjelaskan data `y_pred` yang disimpan atau dikirim pada bagian ini.
    y_pred: np.ndarray,
    # Menjelaskan data `ground_truth_threshold` yang disimpan atau dikirim pada bagian ini.
    ground_truth_threshold: float,
    # Menjelaskan data `threshold_step` yang disimpan atau dikirim pada bagian ini.
    threshold_step: float,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    # Mengecek syarat sebelum melanjutkan proses.
    if threshold_step <= 0 or threshold_step >= 1:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("threshold_step harus berada pada rentang (0, 1).")

    # Menyimpan nilai ke `sweep_values` untuk dipakai pada langkah berikutnya.
    sweep_values = [round(value, 4) for value in np.arange(threshold_step, 1.0, threshold_step)]
    # Menyimpan nilai ke `results` untuk dipakai pada langkah berikutnya.
    results: list[dict[str, float]] = []
    # Menyimpan nilai ke `best_threshold` untuk dipakai pada langkah berikutnya.
    best_threshold = 0.5
    # Menyimpan nilai ke `best_metrics` untuk dipakai pada langkah berikutnya.
    best_metrics: dict[str, float] | None = None

    # Mengulang proses untuk setiap data dalam daftar.
    for threshold in sweep_values:
        # Menyimpan nilai ke `metrics` untuk dipakai pada langkah berikutnya.
        metrics = compute_metrics(y_true, y_pred, threshold, ground_truth_threshold)
        # Melanjutkan langkah kerja pada bagian kode ini.
        results.append({"threshold": threshold, **metrics})
        # Mengecek syarat sebelum melanjutkan proses.
        if best_metrics is None:
            # Menyimpan nilai ke `best_threshold` untuk dipakai pada langkah berikutnya.
            best_threshold = threshold
            # Menyimpan nilai ke `best_metrics` untuk dipakai pada langkah berikutnya.
            best_metrics = metrics
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue

        # Menyimpan nilai ke `candidate_key` untuk dipakai pada langkah berikutnya.
        candidate_key = (
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
        # Menyimpan nilai ke `current_key` untuk dipakai pada langkah berikutnya.
        current_key = (
            # Melanjutkan langkah kerja pada bagian kode ini.
            best_metrics["f1_score"],
            # Melanjutkan langkah kerja pada bagian kode ini.
            best_metrics["iou"],
            # Melanjutkan langkah kerja pada bagian kode ini.
            best_metrics["precision"],
            # Melanjutkan langkah kerja pada bagian kode ini.
            best_metrics["recall"],
            # Melanjutkan langkah kerja pada bagian kode ini.
            -best_metrics["fp"],
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Mengecek syarat sebelum melanjutkan proses.
        if candidate_key > current_key:
            # Menyimpan nilai ke `best_threshold` untuk dipakai pada langkah berikutnya.
            best_threshold = threshold
            # Menyimpan nilai ke `best_metrics` untuk dipakai pada langkah berikutnya.
            best_metrics = metrics

    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert best_metrics is not None
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return best_threshold, results, best_metrics


# Membuat langkah kerja bernama `build_synthetic_demo`.
def build_synthetic_demo() -> tuple[np.ndarray, np.ndarray]:
    # Menyimpan nilai ke `y_true` untuk dipakai pada langkah berikutnya.
    y_true = np.zeros((1, 32, 32, 1), dtype=np.float32)
    # Menyimpan nilai ke `y_true[0, 10` untuk dipakai pada langkah berikutnya.
    y_true[0, 10:13, 11:14, 0] = 1.0
    # Menyimpan nilai ke `y_true[0, 20` untuk dipakai pada langkah berikutnya.
    y_true[0, 20:23, 22:25, 0] = 0.8

    # Menyimpan nilai ke `y_pred` untuk dipakai pada langkah berikutnya.
    y_pred = np.zeros((1, 32, 32, 1), dtype=np.float32)
    # Menyimpan nilai ke `y_pred[0, 11` untuk dipakai pada langkah berikutnya.
    y_pred[0, 11:15, 12:16, 0] = 0.72
    # Menyimpan nilai ke `y_pred[0, 19` untuk dipakai pada langkah berikutnya.
    y_pred[0, 19:24, 21:26, 0] = 0.58
    # Menyimpan nilai ke `y_pred[0, 4` untuk dipakai pada langkah berikutnya.
    y_pred[0, 4:6, 5:7, 0] = 0.42
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return y_true, y_pred


# Membuat langkah kerja bernama `build_parser`.
def build_parser() -> argparse.ArgumentParser:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description=(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Ringkasan fase Evaluation historical risk patch. "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Script ini fokus pada threshold sweep, standard metrics, dan buffered evaluation."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--report-json",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="backend/models/historical_risk_patch_20260416/training_report_best_1epoch.json",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Report training yang ingin diringkas.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--probability-npy", default="", help="Path probability map .npy untuk evaluasi array nyata.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--ground-truth-image", default="", help="Path gambar target aktual untuk evaluasi array nyata.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--native-width", type=int, default=1528)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--native-height", type=int, default=773)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--label-dilation-kernel", type=int, default=9)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--label-blur-radius", type=float, default=2.0)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--ground-truth-threshold", type=float, default=0.05)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--threshold-step", type=float, default=0.05)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--buffer-radius", type=int, default=5)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--json", action="store_true")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser


# Membuat langkah kerja bernama `summarize_report`.
def summarize_report(report_path: Path) -> dict:
    # Menyimpan nilai ke `data` untuk dipakai pada langkah berikutnya.
    data = json.loads(report_path.read_text(encoding="utf-8"))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "mode": "training_report",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "report_path": str(report_path),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_profile": "historical_risk_patch_160",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "loss_strategy": data.get("loss_strategy"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "selected_threshold": data.get("selected_threshold"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "evaluation_buffer_radius": data.get("evaluation_buffer_radius"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "validation_best_metrics": data.get("validation_best_metrics"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "validation_buffered_metrics_best_threshold": data.get("validation_buffered_metrics_best_threshold"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_metrics_best_threshold": data.get("test_metrics_best_threshold"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_buffered_metrics_best_threshold": data.get("test_buffered_metrics_best_threshold"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "default_threshold": data.get("default_threshold"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_metrics_default_threshold": data.get("test_metrics_default_threshold"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_buffered_metrics_default_threshold": data.get("test_buffered_metrics_default_threshold"),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }


# Membuat langkah kerja bernama `summarize_array_evaluation`.
def summarize_array_evaluation(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `probability_path` untuk dipakai pada langkah berikutnya.
    probability_path = Path(args.probability_npy).expanduser().resolve()
    # Menyimpan nilai ke `ground_truth_path` untuk dipakai pada langkah berikutnya.
    ground_truth_path = Path(args.ground_truth_image).expanduser().resolve()
    # Mengecek syarat sebelum melanjutkan proses.
    if not probability_path.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError(f"Probability array tidak ditemukan: {probability_path}")
    # Mengecek syarat sebelum melanjutkan proses.
    if not ground_truth_path.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError(f"Gambar ground truth tidak ditemukan: {ground_truth_path}")

    # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
    probability = np.load(probability_path).astype(np.float32)
    # Mengecek syarat sebelum melanjutkan proses.
    if probability.ndim == 2:
        # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
        probability = probability[None, :, :, None]
    # Mengecek syarat lain jika syarat sebelumnya tidak terpenuhi.
    elif probability.ndim == 3:
        # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
        probability = probability[:, :, :, None]

    # Menyimpan nilai ke `ground_truth` untuk dipakai pada langkah berikutnya.
    ground_truth = load_native_risk_map(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        ground_truth_path,
        # Menyimpan nilai ke `native_width` untuk dipakai pada langkah berikutnya.
        native_width=args.native_width,
        # Menyimpan nilai ke `native_height` untuk dipakai pada langkah berikutnya.
        native_height=args.native_height,
        # Menyimpan nilai ke `label_dilation_kernel` untuk dipakai pada langkah berikutnya.
        label_dilation_kernel=args.label_dilation_kernel,
        # Menyimpan nilai ke `label_blur_radius` untuk dipakai pada langkah berikutnya.
        label_blur_radius=args.label_blur_radius,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `ground_truth` untuk dipakai pada langkah berikutnya.
    ground_truth = ground_truth[None, :, :, None]

    # Menyimpan nilai ke `best_threshold, sweep_results, best_metrics` untuk dipakai pada langkah berikutnya.
    best_threshold, sweep_results, best_metrics = sweep_thresholds(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        ground_truth,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        probability,
        # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        ground_truth_threshold=args.ground_truth_threshold,
        # Menyimpan nilai ke `threshold_step` untuk dipakai pada langkah berikutnya.
        threshold_step=args.threshold_step,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `buffered_metrics` untuk dipakai pada langkah berikutnya.
    buffered_metrics = compute_buffered_metrics(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        ground_truth,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        probability,
        # Menyimpan nilai ke `threshold` untuk dipakai pada langkah berikutnya.
        threshold=best_threshold,
        # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        ground_truth_threshold=args.ground_truth_threshold,
        # Menyimpan nilai ke `buffer_radius` untuk dipakai pada langkah berikutnya.
        buffer_radius=args.buffer_radius,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "mode": "real_array_evaluation",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "probability_npy": str(probability_path),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "ground_truth_image": str(ground_truth_path),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "selected_threshold": best_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "ground_truth_threshold": args.ground_truth_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "buffer_radius": args.buffer_radius,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "best_metrics": best_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "buffered_metrics": buffered_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "threshold_sweep_preview": sweep_results[:5],
        # Melanjutkan langkah kerja pada bagian kode ini.
        "output_shape": list(probability.shape),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }


# Membuat langkah kerja bernama `summarize_synthetic_demo`.
def summarize_synthetic_demo(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `y_true, y_pred` untuk dipakai pada langkah berikutnya.
    y_true, y_pred = build_synthetic_demo()
    # Menyimpan nilai ke `best_threshold, sweep_results, best_metrics` untuk dipakai pada langkah berikutnya.
    best_threshold, sweep_results, best_metrics = sweep_thresholds(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_true,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_pred,
        # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        ground_truth_threshold=args.ground_truth_threshold,
        # Menyimpan nilai ke `threshold_step` untuk dipakai pada langkah berikutnya.
        threshold_step=args.threshold_step,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `buffered_metrics` untuk dipakai pada langkah berikutnya.
    buffered_metrics = compute_buffered_metrics(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_true,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_pred,
        # Menyimpan nilai ke `threshold` untuk dipakai pada langkah berikutnya.
        threshold=best_threshold,
        # Menyimpan nilai ke `ground_truth_threshold` untuk dipakai pada langkah berikutnya.
        ground_truth_threshold=args.ground_truth_threshold,
        # Menyimpan nilai ke `buffer_radius` untuk dipakai pada langkah berikutnya.
        buffer_radius=args.buffer_radius,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "mode": "synthetic_demo",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "selected_threshold": best_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "ground_truth_threshold": args.ground_truth_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "buffer_radius": args.buffer_radius,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "best_metrics": best_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "buffered_metrics": buffered_metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "threshold_sweep_preview": sweep_results[:5],
        # Melanjutkan langkah kerja pada bagian kode ini.
        "output_shape": list(y_pred.shape),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }


# Membuat langkah kerja bernama `build_summary`.
def build_summary(args: argparse.Namespace) -> dict:
    # Mengecek syarat sebelum melanjutkan proses.
    if args.probability_npy and args.ground_truth_image:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return summarize_array_evaluation(args)

    # Menyimpan nilai ke `report_path` untuk dipakai pada langkah berikutnya.
    report_path = Path(args.report_json).expanduser().resolve()
    # Mengecek syarat sebelum melanjutkan proses.
    if report_path.exists():
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return summarize_report(report_path)

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return summarize_synthetic_demo(args)


# Membuat langkah kerja bernama `_print_header`.
def _print_header(title: str) -> None:
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print()
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("=" * 72)
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(title)
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("=" * 72)


# Membuat langkah kerja bernama `_print_key_value`.
def _print_key_value(label: str, value) -> None:
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"{label:<24}: {value}")


# Membuat langkah kerja bernama `_format_metric_value`.
def _format_metric_value(key: str, value) -> str:
    # Mengecek syarat sebelum melanjutkan proses.
    if value is None:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "-"
    # Mengecek syarat sebelum melanjutkan proses.
    if key in {"tp", "fp", "fn", "tn", "buffer_radius"}:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"{int(value):,}"
    # Mengecek syarat sebelum melanjutkan proses.
    if isinstance(value, float):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"{value:.4f}"
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return str(value)


# Membuat langkah kerja bernama `_print_metrics_block`.
def _print_metrics_block(title: str, metrics: dict | None) -> None:
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_header(title)
    # Mengecek syarat sebelum melanjutkan proses.
    if not metrics:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("Tidak ada data metrik.")
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    # Menyimpan nilai ke `ordered_keys` untuk dipakai pada langkah berikutnya.
    ordered_keys = ["tp", "fp", "fn", "tn", "precision", "recall", "f1_score", "iou", "accuracy", "buffer_radius"]
    # Menyimpan nilai ke `label_map` untuk dipakai pada langkah berikutnya.
    label_map = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tp": "TP",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "fp": "FP",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "fn": "FN",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tn": "TN",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "precision": "Precision",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recall": "Recall",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "f1_score": "F1-Score",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "iou": "IoU",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "accuracy": "Accuracy",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "buffer_radius": "Buffer Radius",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }

    # Mengulang proses untuk setiap data dalam daftar.
    for key in ordered_keys:
        # Mengecek syarat sebelum melanjutkan proses.
        if key not in metrics:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"{label_map[key]:<24}: {_format_metric_value(key, metrics[key])}")


# Membuat langkah kerja bernama `_print_threshold_sweep_table`.
def _print_threshold_sweep_table(results: list[dict]) -> None:
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_header("THRESHOLD SWEEP PREVIEW")
    # Mengecek syarat sebelum melanjutkan proses.
    if not results:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("Tidak ada data threshold sweep.")
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'IoU':<12}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("-" * 60)
    # Mengulang proses untuk setiap data dalam daftar.
    for item in results:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{item['threshold']:<12.4f}"
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{item['precision']:<12.4f}"
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{item['recall']:<12.4f}"
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{item['f1_score']:<12.4f}"
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{item['iou']:<12.4f}"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )


# Membuat langkah kerja bernama `print_human_summary`.
def print_human_summary(summary: dict) -> None:
    # Menyimpan nilai ke `mode` untuk dipakai pada langkah berikutnya.
    mode = summary["mode"]
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_header("RINGKASAN EVALUATION")
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_key_value("Mode", mode)

    # Mengecek syarat sebelum melanjutkan proses.
    if mode == "training_report":
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_key_value("Model profile", summary["model_profile"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_key_value("Report", summary["report_path"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_key_value("Loss strategy", summary["loss_strategy"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_key_value("Threshold terpilih", summary["selected_threshold"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_key_value("Threshold default", summary["default_threshold"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_key_value("Buffer radius", summary["evaluation_buffer_radius"])

        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_metrics_block("VALIDATION - BEST THRESHOLD", summary["validation_best_metrics"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_metrics_block("VALIDATION - BUFFERED", summary["validation_buffered_metrics_best_threshold"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_metrics_block("TEST - BEST THRESHOLD", summary["test_metrics_best_threshold"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_metrics_block("TEST - BUFFERED BEST THRESHOLD", summary["test_buffered_metrics_best_threshold"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_metrics_block("TEST - DEFAULT THRESHOLD", summary["test_metrics_default_threshold"])
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        _print_metrics_block(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "TEST - BUFFERED DEFAULT THRESHOLD",
            # Melanjutkan langkah kerja pada bagian kode ini.
            summary["test_buffered_metrics_default_threshold"],
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_key_value("Threshold terpilih", f"{summary['selected_threshold']:.4f}")
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_key_value("Ground truth threshold", summary["ground_truth_threshold"])
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_key_value("Buffer radius", summary["buffer_radius"])
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_key_value("Output shape", summary["output_shape"])

    # Mengecek syarat sebelum melanjutkan proses.
    if mode == "real_array_evaluation":
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_key_value("Probability NPY", summary["probability_npy"])
        # Melanjutkan langkah kerja pada bagian kode ini.
        _print_key_value("Ground truth image", summary["ground_truth_image"])

    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_metrics_block("STANDARD METRICS", summary["best_metrics"])
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_metrics_block("BUFFERED METRICS", summary["buffered_metrics"])
    # Melanjutkan langkah kerja pada bagian kode ini.
    _print_threshold_sweep_table(summary["threshold_sweep_preview"])


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
        print(json.dumps(summary, indent=2, ensure_ascii=False))


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Melanjutkan langkah kerja pada bagian kode ini.
    main()
