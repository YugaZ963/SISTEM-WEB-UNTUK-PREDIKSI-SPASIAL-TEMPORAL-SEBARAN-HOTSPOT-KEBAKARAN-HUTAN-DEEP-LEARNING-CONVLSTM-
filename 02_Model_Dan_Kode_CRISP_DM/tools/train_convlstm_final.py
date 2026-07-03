
"""Komentar file skripsi:
Script training awal ConvLSTM sebelum pipeline historical risk patch dikembangkan.

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
# TensorFlow dipakai untuk membangun, melatih, memuat, dan menjalankan model ConvLSTM.
import tensorflow as tf
# Pillow Image dipakai untuk membuka citra hotspot PNG/JPG dan menyimpan preview/hasil.
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
from tensorflow.keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Dropout, Input


# Regex ini mengambil tanggal dari nama file dataset FIRMS agar urutan temporal dapat dibentuk.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")


# Decorator ini membuat record dataset/statistik lebih ringkas tanpa menulis constructor manual.
@dataclass(frozen=True)
# Class ini menyatukan path citra hotspot dan tanggalnya agar urutan temporal tidak tertukar.
class DatasetRecord:
    path: Path
    date: datetime


# Membaca semua parameter training seperti dataset, patch, threshold, batch, epoch, dan lokasi output.
def parse_args() -> argparse.Namespace:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description="Latih model ConvLSTM final dari dataset hotspot harian lokal."
    )
    parser.add_argument(
        "--dataset-dir",
        default="Ipynb/Dataset History Fire Hotspot In Riau Province",
        help="Folder dataset JPG harian.",
    )
    parser.add_argument(
        "--output-model",
        default="Ipynb/best_model_convlstm.keras",
        help="Path model terbaik output.",
    )
    parser.add_argument(
        "--output-report",
        default="Ipynb/convlstm_training_report.json",
        help="Path JSON ringkasan training.",
    )
    # Opsi `--img-size` menambah parameter eksekusi script.
    parser.add_argument("--img-size", type=int, default=128, help="Ukuran citra grid.")
    # Opsi ini menentukan jumlah frame historis; project memakai 7 frame H-6 sampai H0.
    parser.add_argument("--seq-length", type=int, default=7, help="Panjang sequence input.")
    # Opsi ini mengatur jumlah putaran training model.
    parser.add_argument("--epochs", type=int, default=20, help="Jumlah epoch maksimum.")
    # Opsi ini mengatur jumlah patch/sequence yang diproses sekali training.
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size training.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Rasio train kronologis.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Rasio validation kronologis.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold evaluasi mask prediksi.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa model/report jika sudah ada.",
    )
    # Hasil ini dikembalikan sebagai output fungsi `parse_args` untuk tahap berikutnya.
    return parser.parse_args()


# Fungsi `_extract_hotspot_mask` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _extract_hotspot_mask(path: Path, img_size: int) -> np.ndarray:
    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    with Image.open(path) as image:
        image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        # Representasi HSV memudahkan pemisahan warna merah hotspot dibanding RGB langsung.
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)

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
    mask = (red_low | red_high).astype(np.float32)
    # Hasil ini dikembalikan sebagai output fungsi `_extract_hotspot_mask` untuk tahap berikutnya.
    return np.expand_dims(mask, axis=-1)


# Membaca file citra hotspot dari folder dataset dan mengurutkannya berdasarkan tanggal pada nama file.
def load_records(dataset_dir: Path) -> list[DatasetRecord]:
    # Daftar record citra hotspot yang sudah terbaca dari folder dataset.
    records: list[DatasetRecord] = []
    for path in sorted(dataset_dir.glob("*.jpg")):
        # Nama file diperiksa agar tanggal FIRMS bisa dipakai sebagai urutan waktu.
        match = DATE_PATTERN.search(path.name)
        if not match:
            continue
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        records.append(
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            DatasetRecord(
                path=path,
                date=datetime.strptime(match.group(1), "%Y-%m-%d"),
            )
        )
    # Hasil ini dikembalikan sebagai output fungsi `load_records` untuk tahap berikutnya.
    return sorted(records, key=lambda item: item.date)


# Fungsi `build_sequences` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def build_sequences(images: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    x_data: list[np.ndarray] = []
    y_data: list[np.ndarray] = []
    for index in range(len(images) - seq_length):
        x_data.append(images[index : index + seq_length])
        y_data.append(images[index + seq_length])
    # Hasil ini dikembalikan sebagai output fungsi `build_sequences` untuk tahap berikutnya.
    return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)


# Menyusun arsitektur ConvLSTM yang menerima sequence 7 frame dan menghasilkan peta probabilitas H+1.
def build_model(seq_length: int, img_size: int) -> tf.keras.Model:
    model = Sequential(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            Input(shape=(seq_length, img_size, img_size, 1)),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Hasil ini dikembalikan sebagai output fungsi `build_model` untuk tahap berikutnya.
    return model


# Fungsi `compute_metrics` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    truth = y_true >= threshold
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


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> int:
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parse_args()
    # `dataset_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    dataset_dir = Path(args.dataset_dir).resolve()
    output_model = Path(args.output_model).resolve()
    output_report = Path(args.output_report).resolve()

    # Validasi ini menghentikan proses jika folder dataset hotspot belum tersedia.
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_dir}")

    if output_model.exists() and not args.overwrite:
        raise FileExistsError(f"Model sudah ada: {output_model}. Gunakan --overwrite.")

    if output_report.exists() and not args.overwrite:
        raise FileExistsError(f"Report sudah ada: {output_report}. Gunakan --overwrite.")

    # Daftar record citra hotspot yang sudah terbaca dari folder dataset.
    records = load_records(dataset_dir)
    # Pengecekan ini memastikan jumlah citra cukup untuk membentuk sequence tujuh hari dan target H+1.
    if len(records) <= args.seq_length:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Jumlah data tidak cukup untuk membentuk sequence.")

    print(f"Dataset ditemukan: {len(records)} frame")
    print(f"Rentang data: {records[0].date.date()} s.d. {records[-1].date.date()}")

    images = np.stack(
        [_extract_hotspot_mask(record.path, img_size=args.img_size) for record in records],
        axis=0,
    ).astype(np.float32)

    x_data, y_data = build_sequences(images, seq_length=args.seq_length)
    sample_count = len(x_data)
    if sample_count < 3:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Jumlah sample sequence terlalu sedikit.")

    train_end = max(1, int(sample_count * args.train_ratio))
    val_end = max(train_end + 1, int(sample_count * (args.train_ratio + args.val_ratio)))
    val_end = min(val_end, sample_count - 1)

    x_train, y_train = x_data[:train_end], y_data[:train_end]
    x_val, y_val = x_data[train_end:val_end], y_data[train_end:val_end]
    x_test, y_test = x_data[val_end:], y_data[val_end:]

    if len(x_val) == 0 or len(x_test) == 0:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("Split train/val/test gagal. Periksa rasio dataset.")

    print(f"Train samples: {len(x_train)}")
    print(f"Val samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

    model = build_model(seq_length=args.seq_length, img_size=args.img_size)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        ModelCheckpoint(
            filepath=str(output_model),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Training dijalankan pada sequence/patch yang sudah disiapkan dari citra hotspot historis.
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Model menghasilkan probability map risiko hotspot dari input sequence.
    predictions = model.predict(x_test, verbose=0)
    # Nilai precision, recall, F1, IoU, dan metrik lain hasil evaluasi.
    metrics = compute_metrics(y_true=y_test, y_pred=predictions, threshold=args.threshold)

    report = {
        "dataset_dir": str(dataset_dir),
        "dataset_frame_count": len(records),
        "date_start": records[0].date.strftime("%Y-%m-%d"),
        "date_end": records[-1].date.strftime("%Y-%m-%d"),
        "img_size": args.img_size,
        "seq_length": args.seq_length,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "test_samples": len(x_test),
        "threshold": args.threshold,
        "history": history.history,
        "test_metrics": metrics,
        "output_model": str(output_model),
    }

    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"Model terbaik tersimpan di: {output_model}")
    print(f"Report training tersimpan di: {output_report}")
    # Hasil ini dikembalikan sebagai output fungsi `main` untuk tahap berikutnya.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
