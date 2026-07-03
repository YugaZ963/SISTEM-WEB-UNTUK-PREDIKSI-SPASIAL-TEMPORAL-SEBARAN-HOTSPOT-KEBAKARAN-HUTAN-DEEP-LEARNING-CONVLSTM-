
"""Komentar file skripsi:
Script ringkas fase Evaluation CRISP-DM untuk membaca report training dan menampilkan metrik model.

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
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path

# NumPy dipakai untuk mask hotspot, tensor sequence, patch, probability map, dan metrik piksel.
import numpy as np
# Pillow Image dipakai untuk membuka citra hotspot PNG/JPG dan menyimpan preview/hasil.
from PIL import Image, ImageFilter


# Regex ini mengambil tanggal dari nama file dataset FIRMS agar urutan temporal dapat dibentuk.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")


# Memastikan ukuran kernel dilation/blur bersifat ganjil karena filter gambar membutuhkan pusat kernel.
def _normalize_kernel(size: int) -> int:
    size = max(1, int(size))
    # Hasil ini dikembalikan sebagai output fungsi `_normalize_kernel` untuk tahap berikutnya.
    return size if size % 2 == 1 else size + 1


# Membentuk target risk map H+1 dari hotspot merah, dilation label, dan Gaussian blur.
def load_native_risk_map(
    path: Path,
    native_width: int,
    native_height: int,
    label_dilation_kernel: int,
    label_blur_radius: float,
) -> np.ndarray:
    label_dilation_kernel = _normalize_kernel(label_dilation_kernel)
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


# Fungsi `compute_metrics` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, ground_truth_threshold: float) -> dict[str, float]:
    truth = y_true >= ground_truth_threshold
    pred = y_pred >= threshold
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())

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


# Fungsi `_dilate_binary_array` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _dilate_binary_array(mask: np.ndarray, buffer_radius: int) -> np.ndarray:
    if buffer_radius <= 0:
        # Hasil ini dikembalikan sebagai output fungsi `_dilate_binary_array` untuk tahap berikutnya.
        return mask.astype(bool)

    original_shape = mask.shape
    if mask.ndim == 4 and mask.shape[-1] == 1:
        squeezed = np.squeeze(mask.astype(np.uint8), axis=-1)
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        squeezed = mask.astype(np.uint8)

    if squeezed.ndim == 2:
        flat = squeezed.reshape(1, squeezed.shape[0], squeezed.shape[1])
    elif squeezed.ndim == 3:
        flat = squeezed.reshape(-1, squeezed.shape[-2], squeezed.shape[-1])
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError(f"Shape mask tidak didukung untuk buffered evaluation: {mask.shape}")

    kernel_size = _normalize_kernel((int(buffer_radius) * 2) + 1)
    dilated: list[np.ndarray] = []
    for item in flat:
        image = Image.fromarray((item > 0).astype(np.uint8) * 255)
        # MaxFilter memperluas area positif sehingga hotspot kecil tidak hilang saat dipatch.
        image = image.filter(ImageFilter.MaxFilter(size=kernel_size))
        dilated.append(np.asarray(image, dtype=np.uint8) > 0)

    stacked = np.stack(dilated, axis=0)
    if squeezed.ndim == 2:
        result = stacked[0]
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        result = stacked.reshape(squeezed.shape)

    if len(original_shape) == 4 and original_shape[-1] == 1:
        result = np.expand_dims(result, axis=-1)
    # Hasil ini dikembalikan sebagai output fungsi `_dilate_binary_array` untuk tahap berikutnya.
    return result.astype(bool)


# Fungsi `compute_buffered_metrics` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def compute_buffered_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    ground_truth_threshold: float,
    buffer_radius: int,
) -> dict[str, float]:
    truth = y_true >= ground_truth_threshold
    pred = y_pred >= threshold
    truth = _dilate_binary_array(truth, buffer_radius)
    pred = _dilate_binary_array(pred, buffer_radius)

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())

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
        "buffer_radius": int(buffer_radius),
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


# Fungsi `sweep_thresholds` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def sweep_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ground_truth_threshold: float,
    threshold_step: float,
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    if threshold_step <= 0 or threshold_step >= 1:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("threshold_step harus berada pada rentang (0, 1).")

    sweep_values = [round(value, 4) for value in np.arange(threshold_step, 1.0, threshold_step)]
    results: list[dict[str, float]] = []
    # `best_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
    best_threshold = 0.5
    # `best_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    best_metrics: dict[str, float] | None = None

    # Loop ini mencoba beberapa threshold untuk mencari ambang evaluasi terbaik.
    for threshold in sweep_values:
        # Nilai precision, recall, F1, IoU, dan metrik lain hasil evaluasi.
        metrics = compute_metrics(y_true, y_pred, threshold, ground_truth_threshold)
        results.append({"threshold": threshold, **metrics})
        if best_metrics is None:
            # `best_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
            best_threshold = threshold
            # `best_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
            best_metrics = metrics
            continue

        candidate_key = (
            metrics["f1_score"],
            metrics["iou"],
            metrics["precision"],
            metrics["recall"],
            -metrics["fp"],
        )
        current_key = (
            best_metrics["f1_score"],
            best_metrics["iou"],
            best_metrics["precision"],
            best_metrics["recall"],
            -best_metrics["fp"],
        )
        if candidate_key > current_key:
            # `best_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
            best_threshold = threshold
            # `best_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
            best_metrics = metrics

    assert best_metrics is not None
    # Hasil ini dikembalikan sebagai output fungsi `sweep_thresholds` untuk tahap berikutnya.
    return best_threshold, results, best_metrics


# Fungsi `build_synthetic_demo` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def build_synthetic_demo() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.zeros((1, 32, 32, 1), dtype=np.float32)
    y_true[0, 10:13, 11:14, 0] = 1.0
    y_true[0, 20:23, 22:25, 0] = 0.8

    y_pred = np.zeros((1, 32, 32, 1), dtype=np.float32)
    y_pred[0, 11:15, 12:16, 0] = 0.72
    y_pred[0, 19:24, 21:26, 0] = 0.58
    y_pred[0, 4:6, 5:7, 0] = 0.42
    # Hasil ini dikembalikan sebagai output fungsi `build_synthetic_demo` untuk tahap berikutnya.
    return y_true, y_pred


# Menyusun opsi command line agar fase CRISP-DM bisa dijalankan ulang dengan parameter yang jelas.
def build_parser() -> argparse.ArgumentParser:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description=(
            "Ringkasan fase Evaluation historical risk patch. "
            "Script ini fokus pada threshold sweep, standard metrics, dan buffered evaluation."
        )
    )
    parser.add_argument(
        "--report-json",
        default="backend/models/historical_risk_patch_20260416/training_report_best_1epoch.json",
        help="Report training yang ingin diringkas.",
    )
    # Opsi `--probability-npy` menambah parameter eksekusi script.
    parser.add_argument("--probability-npy", default="", help="Path probability map .npy untuk evaluasi array nyata.")
    # Opsi `--ground-truth-image` menambah parameter eksekusi script.
    parser.add_argument("--ground-truth-image", default="", help="Path gambar target aktual untuk evaluasi array nyata.")
    # Opsi ini menyimpan lebar asli peta/citra project.
    parser.add_argument("--native-width", type=int, default=1528)
    # Opsi ini menyimpan tinggi asli peta/citra project.
    parser.add_argument("--native-height", type=int, default=773)
    # Opsi ini memperluas label hotspot agar target H+1 tidak terlalu kecil.
    parser.add_argument("--label-dilation-kernel", type=int, default=9)
    # Opsi ini menghaluskan target risk map supaya model belajar area risiko, bukan titik kaku.
    parser.add_argument("--label-blur-radius", type=float, default=2.0)
    # Opsi ini menentukan batas target risk map yang dihitung sebagai positif.
    parser.add_argument("--ground-truth-threshold", type=float, default=0.05)
    # Opsi `--threshold-step` menambah parameter eksekusi script.
    parser.add_argument("--threshold-step", type=float, default=0.05)
    # Opsi `--buffer-radius` menambah parameter eksekusi script.
    parser.add_argument("--buffer-radius", type=int, default=5)
    # Opsi ini membuat output ringkasan dicetak sebagai JSON agar mudah dikutip/dibaca ulang.
    parser.add_argument("--json", action="store_true")
    # Hasil ini dikembalikan sebagai output fungsi `build_parser` untuk tahap berikutnya.
    return parser


# Fungsi `summarize_report` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def summarize_report(report_path: Path) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    # Hasil ini dikembalikan sebagai output fungsi `summarize_report` untuk tahap berikutnya.
    return {
        "mode": "training_report",
        "report_path": str(report_path),
        "model_profile": "historical_risk_patch_160",
        "loss_strategy": data.get("loss_strategy"),
        "selected_threshold": data.get("selected_threshold"),
        "evaluation_buffer_radius": data.get("evaluation_buffer_radius"),
        "validation_best_metrics": data.get("validation_best_metrics"),
        "validation_buffered_metrics_best_threshold": data.get("validation_buffered_metrics_best_threshold"),
        "test_metrics_best_threshold": data.get("test_metrics_best_threshold"),
        "test_buffered_metrics_best_threshold": data.get("test_buffered_metrics_best_threshold"),
        "default_threshold": data.get("default_threshold"),
        "test_metrics_default_threshold": data.get("test_metrics_default_threshold"),
        "test_buffered_metrics_default_threshold": data.get("test_buffered_metrics_default_threshold"),
    }


# Fungsi `summarize_array_evaluation` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def summarize_array_evaluation(args: argparse.Namespace) -> dict:
    # `probability_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    probability_path = Path(args.probability_npy).expanduser().resolve()
    # `ground_truth_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    ground_truth_path = Path(args.ground_truth_image).expanduser().resolve()
    if not probability_path.exists():
        raise FileNotFoundError(f"Probability array tidak ditemukan: {probability_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Gambar ground truth tidak ditemukan: {ground_truth_path}")

    # Peta probabilitas risiko hotspot dengan nilai 0 sampai 1.
    probability = np.load(probability_path).astype(np.float32)
    if probability.ndim == 2:
        # Peta probabilitas risiko hotspot dengan nilai 0 sampai 1.
        probability = probability[None, :, :, None]
    elif probability.ndim == 3:
        # Peta probabilitas risiko hotspot dengan nilai 0 sampai 1.
        probability = probability[:, :, :, None]

    ground_truth = load_native_risk_map(
        ground_truth_path,
        native_width=args.native_width,
        native_height=args.native_height,
        label_dilation_kernel=args.label_dilation_kernel,
        label_blur_radius=args.label_blur_radius,
    )
    ground_truth = ground_truth[None, :, :, None]

    best_threshold, sweep_results, best_metrics = sweep_thresholds(
        ground_truth,
        probability,
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        ground_truth_threshold=args.ground_truth_threshold,
        # `threshold_step` berkaitan dengan ambang untuk membedakan area risiko dan background.
        threshold_step=args.threshold_step,
    )
    # `buffered_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    buffered_metrics = compute_buffered_metrics(
        ground_truth,
        probability,
        # Ambang untuk mengubah probability map menjadi binary mask area risiko.
        threshold=best_threshold,
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        ground_truth_threshold=args.ground_truth_threshold,
        buffer_radius=args.buffer_radius,
    )
    # Hasil ini dikembalikan sebagai output fungsi `summarize_array_evaluation` untuk tahap berikutnya.
    return {
        "mode": "real_array_evaluation",
        "probability_npy": str(probability_path),
        "ground_truth_image": str(ground_truth_path),
        "selected_threshold": best_threshold,
        "ground_truth_threshold": args.ground_truth_threshold,
        "buffer_radius": args.buffer_radius,
        "best_metrics": best_metrics,
        "buffered_metrics": buffered_metrics,
        "threshold_sweep_preview": sweep_results[:5],
        "output_shape": list(probability.shape),
    }


# Fungsi `summarize_synthetic_demo` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def summarize_synthetic_demo(args: argparse.Namespace) -> dict:
    y_true, y_pred = build_synthetic_demo()
    best_threshold, sweep_results, best_metrics = sweep_thresholds(
        y_true,
        y_pred,
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        ground_truth_threshold=args.ground_truth_threshold,
        # `threshold_step` berkaitan dengan ambang untuk membedakan area risiko dan background.
        threshold_step=args.threshold_step,
    )
    # `buffered_metrics` menyimpan ukuran evaluasi performa prediksi hotspot.
    buffered_metrics = compute_buffered_metrics(
        y_true,
        y_pred,
        # Ambang untuk mengubah probability map menjadi binary mask area risiko.
        threshold=best_threshold,
        # `ground_truth_threshold` berkaitan dengan ambang untuk membedakan area risiko dan background.
        ground_truth_threshold=args.ground_truth_threshold,
        buffer_radius=args.buffer_radius,
    )
    # Hasil ini dikembalikan sebagai output fungsi `summarize_synthetic_demo` untuk tahap berikutnya.
    return {
        "mode": "synthetic_demo",
        "selected_threshold": best_threshold,
        "ground_truth_threshold": args.ground_truth_threshold,
        "buffer_radius": args.buffer_radius,
        "best_metrics": best_metrics,
        "buffered_metrics": buffered_metrics,
        "threshold_sweep_preview": sweep_results[:5],
        "output_shape": list(y_pred.shape),
    }


# Menggabungkan hasil proses menjadi ringkasan JSON/console untuk laporan dan verifikasi eksperimen.
def build_summary(args: argparse.Namespace) -> dict:
    if args.probability_npy and args.ground_truth_image:
        # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
        return summarize_array_evaluation(args)

    # `report_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    report_path = Path(args.report_json).expanduser().resolve()
    if report_path.exists():
        # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
        return summarize_report(report_path)

    # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
    return summarize_synthetic_demo(args)


# Fungsi `_print_header` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# Fungsi `_print_key_value` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _print_key_value(label: str, value) -> None:
    print(f"{label:<24}: {value}")


# Fungsi `_format_metric_value` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _format_metric_value(key: str, value) -> str:
    if value is None:
        # Hasil ini dikembalikan sebagai output fungsi `_format_metric_value` untuk tahap berikutnya.
        return "-"
    if key in {"tp", "fp", "fn", "tn", "buffer_radius"}:
        # Hasil ini dikembalikan sebagai output fungsi `_format_metric_value` untuk tahap berikutnya.
        return f"{int(value):,}"
    if isinstance(value, float):
        # Hasil ini dikembalikan sebagai output fungsi `_format_metric_value` untuk tahap berikutnya.
        return f"{value:.4f}"
    # Hasil ini dikembalikan sebagai output fungsi `_format_metric_value` untuk tahap berikutnya.
    return str(value)


# Fungsi `_print_metrics_block` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _print_metrics_block(title: str, metrics: dict | None) -> None:
    _print_header(title)
    if not metrics:
        print("Tidak ada data metrik.")
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    ordered_keys = ["tp", "fp", "fn", "tn", "precision", "recall", "f1_score", "iou", "accuracy", "buffer_radius"]
    label_map = {
        "tp": "TP",
        "fp": "FP",
        "fn": "FN",
        "tn": "TN",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "iou": "IoU",
        "accuracy": "Accuracy",
        "buffer_radius": "Buffer Radius",
    }

    for key in ordered_keys:
        if key not in metrics:
            continue
        print(f"{label_map[key]:<24}: {_format_metric_value(key, metrics[key])}")


# Fungsi `_print_threshold_sweep_table` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _print_threshold_sweep_table(results: list[dict]) -> None:
    _print_header("THRESHOLD SWEEP PREVIEW")
    if not results:
        print("Tidak ada data threshold sweep.")
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'IoU':<12}")
    print("-" * 60)
    for item in results:
        print(
            f"{item['threshold']:<12.4f}"
            f"{item['precision']:<12.4f}"
            f"{item['recall']:<12.4f}"
            f"{item['f1_score']:<12.4f}"
            f"{item['iou']:<12.4f}"
        )


# Fungsi `print_human_summary` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def print_human_summary(summary: dict) -> None:
    mode = summary["mode"]
    _print_header("RINGKASAN EVALUATION")
    _print_key_value("Mode", mode)

    if mode == "training_report":
        _print_key_value("Model profile", summary["model_profile"])
        _print_key_value("Report", summary["report_path"])
        _print_key_value("Loss strategy", summary["loss_strategy"])
        _print_key_value("Threshold terpilih", summary["selected_threshold"])
        _print_key_value("Threshold default", summary["default_threshold"])
        _print_key_value("Buffer radius", summary["evaluation_buffer_radius"])

        _print_metrics_block("VALIDATION - BEST THRESHOLD", summary["validation_best_metrics"])
        _print_metrics_block("VALIDATION - BUFFERED", summary["validation_buffered_metrics_best_threshold"])
        _print_metrics_block("TEST - BEST THRESHOLD", summary["test_metrics_best_threshold"])
        _print_metrics_block("TEST - BUFFERED BEST THRESHOLD", summary["test_buffered_metrics_best_threshold"])
        _print_metrics_block("TEST - DEFAULT THRESHOLD", summary["test_metrics_default_threshold"])
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        _print_metrics_block(
            "TEST - BUFFERED DEFAULT THRESHOLD",
            summary["test_buffered_metrics_default_threshold"],
        )
        # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
        return

    _print_key_value("Threshold terpilih", f"{summary['selected_threshold']:.4f}")
    _print_key_value("Ground truth threshold", summary["ground_truth_threshold"])
    _print_key_value("Buffer radius", summary["buffer_radius"])
    _print_key_value("Output shape", summary["output_shape"])

    if mode == "real_array_evaluation":
        _print_key_value("Probability NPY", summary["probability_npy"])
        _print_key_value("Ground truth image", summary["ground_truth_image"])

    _print_metrics_block("STANDARD METRICS", summary["best_metrics"])
    _print_metrics_block("BUFFERED METRICS", summary["buffered_metrics"])
    _print_threshold_sweep_table(summary["threshold_sweep_preview"])


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
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
