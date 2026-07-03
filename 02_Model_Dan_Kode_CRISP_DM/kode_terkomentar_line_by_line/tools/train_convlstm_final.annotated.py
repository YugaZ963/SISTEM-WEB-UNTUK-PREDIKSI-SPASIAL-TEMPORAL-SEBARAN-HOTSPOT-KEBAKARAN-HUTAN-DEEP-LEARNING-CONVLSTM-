# File anotasi dari `tools/train_convlstm_final.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Script training awal ConvLSTM sebelum pipeline historical risk patch dikembangkan.

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
from dataclasses import dataclass
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from datetime import datetime
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import tensorflow as tf
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras import Sequential
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Dropout, Input


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DATE_PATTERN = re.compile(r"FIRMS_(\d{4}-\d{2}-\d{2})")


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@dataclass(frozen=True)
# Membuat wadah bernama `DatasetRecord` untuk menyimpan data atau aturan kerja.
class DatasetRecord:
    # Menjelaskan data `path` yang disimpan atau dikirim pada bagian ini.
    path: Path
    # Menjelaskan data `date` yang disimpan atau dikirim pada bagian ini.
    date: datetime


# Membuat langkah kerja bernama `parse_args`.
def parse_args() -> argparse.Namespace:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description="Latih model ConvLSTM final dari dataset hotspot harian lokal."
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--dataset-dir",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="Ipynb/Dataset History Fire Hotspot In Riau Province",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Folder dataset JPG harian.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--output-model",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="Ipynb/best_model_convlstm.keras",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Path model terbaik output.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--output-report",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="Ipynb/convlstm_training_report.json",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Path JSON ringkasan training.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--img-size", type=int, default=128, help="Ukuran citra grid.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--seq-length", type=int, default=7, help="Panjang sequence input.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--epochs", type=int, default=20, help="Jumlah epoch maksimum.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size training.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--train-ratio",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=float,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=0.70,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Rasio train kronologis.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--val-ratio",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=float,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=0.15,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Rasio validation kronologis.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--threshold",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=float,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=0.5,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Threshold evaluasi mask prediksi.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--overwrite",
        # Menyimpan nilai ke `action` untuk dipakai pada langkah berikutnya.
        action="store_true",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Timpa model/report jika sudah ada.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser.parse_args()


# Membuat langkah kerja bernama `_extract_hotspot_mask`.
def _extract_hotspot_mask(path: Path, img_size: int) -> np.ndarray:
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with Image.open(path) as image:
        # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
        image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        # Mengubah warna RGB ke HSV agar warna merah hotspot lebih mudah dipisahkan.
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)

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
    mask = (red_low | red_high).astype(np.float32)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.expand_dims(mask, axis=-1)


# Membuat langkah kerja bernama `load_records`.
def load_records(dataset_dir: Path) -> list[DatasetRecord]:
    # Menyimpan nilai ke `records` untuk dipakai pada langkah berikutnya.
    records: list[DatasetRecord] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for path in sorted(dataset_dir.glob("*.jpg")):
        # Menyimpan nilai ke `match` untuk dipakai pada langkah berikutnya.
        match = DATE_PATTERN.search(path.name)
        # Mengecek syarat sebelum melanjutkan proses.
        if not match:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        records.append(
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            DatasetRecord(
                # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
                path=path,
                # Menyimpan nilai ke `date` untuk dipakai pada langkah berikutnya.
                date=datetime.strptime(match.group(1), "%Y-%m-%d"),
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return sorted(records, key=lambda item: item.date)


# Membuat langkah kerja bernama `build_sequences`.
def build_sequences(images: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    # Menyimpan nilai ke `x_data` untuk dipakai pada langkah berikutnya.
    x_data: list[np.ndarray] = []
    # Menyimpan nilai ke `y_data` untuk dipakai pada langkah berikutnya.
    y_data: list[np.ndarray] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for index in range(len(images) - seq_length):
        # Melanjutkan langkah kerja pada bagian kode ini.
        x_data.append(images[index : index + seq_length])
        # Melanjutkan langkah kerja pada bagian kode ini.
        y_data.append(images[index + seq_length])
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)


# Membuat langkah kerja bernama `build_model`.
def build_model(seq_length: int, img_size: int) -> tf.keras.Model:
    # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
    model = Sequential(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            # Menyimpan nilai ke `Input(shape` untuk dipakai pada langkah berikutnya.
            Input(shape=(seq_length, img_size, img_size, 1)),
            # Menyimpan nilai ke `ConvLSTM2D(filters` untuk dipakai pada langkah berikutnya.
            ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True),
            # Melanjutkan langkah kerja pada bagian kode ini.
            BatchNormalization(),
            # Melanjutkan langkah kerja pada bagian kode ini.
            Dropout(0.2),
            # Menyimpan nilai ke `ConvLSTM2D(filters` untuk dipakai pada langkah berikutnya.
            ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=False),
            # Melanjutkan langkah kerja pada bagian kode ini.
            BatchNormalization(),
            # Melanjutkan langkah kerja pada bagian kode ini.
            Dropout(0.2),
            # Menyimpan nilai ke `Conv2D(filters` untuk dipakai pada langkah berikutnya.
            Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"),
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `model.compile(optimizer` untuk dipakai pada langkah berikutnya.
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return model


# Membuat langkah kerja bernama `compute_metrics`.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    # Menyimpan nilai ke `truth` untuk dipakai pada langkah berikutnya.
    truth = y_true >= threshold
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


# Membuat langkah kerja bernama `main`.
def main() -> int:
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parse_args()
    # Menyimpan nilai ke `dataset_dir` untuk dipakai pada langkah berikutnya.
    dataset_dir = Path(args.dataset_dir).resolve()
    # Menyimpan nilai ke `output_model` untuk dipakai pada langkah berikutnya.
    output_model = Path(args.output_model).resolve()
    # Menyimpan nilai ke `output_report` untuk dipakai pada langkah berikutnya.
    output_report = Path(args.output_report).resolve()

    # Mengecek syarat sebelum melanjutkan proses.
    if not dataset_dir.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_dir}")

    # Mengecek syarat sebelum melanjutkan proses.
    if output_model.exists() and not args.overwrite:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileExistsError(f"Model sudah ada: {output_model}. Gunakan --overwrite.")

    # Mengecek syarat sebelum melanjutkan proses.
    if output_report.exists() and not args.overwrite:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileExistsError(f"Report sudah ada: {output_report}. Gunakan --overwrite.")

    # Menyimpan nilai ke `records` untuk dipakai pada langkah berikutnya.
    records = load_records(dataset_dir)
    # Mengecek syarat sebelum melanjutkan proses.
    if len(records) <= args.seq_length:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Jumlah data tidak cukup untuk membentuk sequence.")

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Dataset ditemukan: {len(records)} frame")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Rentang data: {records[0].date.date()} s.d. {records[-1].date.date()}")

    # Menyimpan nilai ke `images` untuk dipakai pada langkah berikutnya.
    images = np.stack(
        # Menyimpan nilai ke `[_extract_hotspot_mask(record.path, img_size` untuk dipakai pada langkah berikutnya.
        [_extract_hotspot_mask(record.path, img_size=args.img_size) for record in records],
        # Menyimpan nilai ke `axis` untuk dipakai pada langkah berikutnya.
        axis=0,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ).astype(np.float32)

    # Menyimpan nilai ke `x_data, y_data` untuk dipakai pada langkah berikutnya.
    x_data, y_data = build_sequences(images, seq_length=args.seq_length)
    # Menyimpan nilai ke `sample_count` untuk dipakai pada langkah berikutnya.
    sample_count = len(x_data)
    # Mengecek syarat sebelum melanjutkan proses.
    if sample_count < 3:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Jumlah sample sequence terlalu sedikit.")

    # Menyimpan nilai ke `train_end` untuk dipakai pada langkah berikutnya.
    train_end = max(1, int(sample_count * args.train_ratio))
    # Menyimpan nilai ke `val_end` untuk dipakai pada langkah berikutnya.
    val_end = max(train_end + 1, int(sample_count * (args.train_ratio + args.val_ratio)))
    # Menyimpan nilai ke `val_end` untuk dipakai pada langkah berikutnya.
    val_end = min(val_end, sample_count - 1)

    # Menyimpan nilai ke `x_train, y_train` untuk dipakai pada langkah berikutnya.
    x_train, y_train = x_data[:train_end], y_data[:train_end]
    # Menyimpan nilai ke `x_val, y_val` untuk dipakai pada langkah berikutnya.
    x_val, y_val = x_data[train_end:val_end], y_data[train_end:val_end]
    # Menyimpan nilai ke `x_test, y_test` untuk dipakai pada langkah berikutnya.
    x_test, y_test = x_data[val_end:], y_data[val_end:]

    # Mengecek syarat sebelum melanjutkan proses.
    if len(x_val) == 0 or len(x_test) == 0:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("Split train/val/test gagal. Periksa rasio dataset.")

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Train samples: {len(x_train)}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Val samples: {len(x_val)}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Test samples: {len(x_test)}")

    # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
    model = build_model(seq_length=args.seq_length, img_size=args.img_size)
    # Menyimpan nilai ke `output_model.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_model.parent.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `output_report.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_report.parent.mkdir(parents=True, exist_ok=True)

    # Menyimpan nilai ke `callbacks` untuk dipakai pada langkah berikutnya.
    callbacks = [
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        ModelCheckpoint(
            # Menyimpan nilai ke `filepath` untuk dipakai pada langkah berikutnya.
            filepath=str(output_model),
            # Menyimpan nilai ke `monitor` untuk dipakai pada langkah berikutnya.
            monitor="val_loss",
            # Menyimpan nilai ke `save_best_only` untuk dipakai pada langkah berikutnya.
            save_best_only=True,
            # Menyimpan nilai ke `mode` untuk dipakai pada langkah berikutnya.
            mode="min",
            # Menyimpan nilai ke `verbose` untuk dipakai pada langkah berikutnya.
            verbose=1,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        EarlyStopping(
            # Menyimpan nilai ke `monitor` untuk dipakai pada langkah berikutnya.
            monitor="val_loss",
            # Menyimpan nilai ke `patience` untuk dipakai pada langkah berikutnya.
            patience=5,
            # Menyimpan nilai ke `restore_best_weights` untuk dipakai pada langkah berikutnya.
            restore_best_weights=True,
            # Menyimpan nilai ke `verbose` untuk dipakai pada langkah berikutnya.
            verbose=1,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        ReduceLROnPlateau(
            # Menyimpan nilai ke `monitor` untuk dipakai pada langkah berikutnya.
            monitor="val_loss",
            # Menyimpan nilai ke `factor` untuk dipakai pada langkah berikutnya.
            factor=0.5,
            # Menyimpan nilai ke `patience` untuk dipakai pada langkah berikutnya.
            patience=2,
            # Menyimpan nilai ke `min_lr` untuk dipakai pada langkah berikutnya.
            min_lr=1e-6,
            # Menyimpan nilai ke `verbose` untuk dipakai pada langkah berikutnya.
            verbose=1,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ]

    # Menyimpan nilai ke `history` untuk dipakai pada langkah berikutnya.
    history = model.fit(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        x_train,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        y_train,
        # Menyimpan nilai ke `validation_data` untuk dipakai pada langkah berikutnya.
        validation_data=(x_val, y_val),
        # Menyimpan nilai ke `epochs` untuk dipakai pada langkah berikutnya.
        epochs=args.epochs,
        # Menyimpan nilai ke `batch_size` untuk dipakai pada langkah berikutnya.
        batch_size=args.batch_size,
        # Menyimpan nilai ke `callbacks` untuk dipakai pada langkah berikutnya.
        callbacks=callbacks,
        # Menyimpan nilai ke `verbose` untuk dipakai pada langkah berikutnya.
        verbose=1,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `predictions` untuk dipakai pada langkah berikutnya.
    predictions = model.predict(x_test, verbose=0)
    # Menyimpan nilai ke `metrics` untuk dipakai pada langkah berikutnya.
    metrics = compute_metrics(y_true=y_test, y_pred=predictions, threshold=args.threshold)

    # Menyimpan nilai ke `report` untuk dipakai pada langkah berikutnya.
    report = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dataset_dir": str(dataset_dir),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dataset_frame_count": len(records),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "date_start": records[0].date.strftime("%Y-%m-%d"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "date_end": records[-1].date.strftime("%Y-%m-%d"),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "img_size": args.img_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "seq_length": args.seq_length,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "epochs_requested": args.epochs,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "batch_size": args.batch_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_ratio": args.train_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_ratio": args.val_ratio,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "train_samples": len(x_train),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "val_samples": len(x_val),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_samples": len(x_test),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "threshold": args.threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "history": history.history,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "test_metrics": metrics,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "output_model": str(output_model),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }

    # Menyimpan nilai ke `output_report.write_text(json.dumps(report, indent` untuk dipakai pada langkah berikutnya.
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(json.dumps(report["test_metrics"], indent=2))
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Model terbaik tersimpan di: {output_model}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Report training tersimpan di: {output_report}")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return 0


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Menghentikan proses dan memberi pesan kesalahan yang jelas.
    raise SystemExit(main())
