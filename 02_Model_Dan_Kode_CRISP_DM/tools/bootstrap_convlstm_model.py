
"""Komentar file skripsi:
Tool bootstrap model awal untuk memastikan file model dan struktur backend tersedia.

Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
terutama data understanding, data preparation, modeling, evaluation,
atau deployment sesuai fungsi teknisnya.
"""

from __future__ import annotations

# argparse dipakai agar parameter dataset, patch, threshold, dan output bisa diatur dari command line.
import argparse
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path

# TensorFlow dipakai untuk membangun, melatih, memuat, dan menjalankan model ConvLSTM.
import tensorflow as tf
from tensorflow.keras import Sequential
# Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
from tensorflow.keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Input


# Menyusun arsitektur ConvLSTM yang menerima sequence 7 frame dan menghasilkan peta probabilitas H+1.
def build_model(img_size: int, seq_length: int) -> tf.keras.Model:
    model = Sequential(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            Input(shape=(seq_length, img_size, img_size, 1)),
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                return_sequences=True,
            ),
            BatchNormalization(),
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                return_sequences=False,
            ),
            BatchNormalization(),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Hasil ini dikembalikan sebagai output fungsi `build_model` untuk tahap berikutnya.
    return model


# Membaca semua parameter training seperti dataset, patch, threshold, batch, epoch, dan lokasi output.
def parse_args() -> argparse.Namespace:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description="Buat model ConvLSTM bootstrap agar integrasi web bisa diuji tanpa menunggu model training final."
    )
    parser.add_argument(
        "--output",
        default="Ipynb/best_model_convlstm.keras",
        help="Path output model .keras/.h5",
    )
    # Opsi `--img-size` menambah parameter eksekusi script.
    parser.add_argument("--img-size", type=int, default=128, help="Ukuran grid citra.")
    # Opsi ini menentukan jumlah frame historis; project memakai 7 frame H-6 sampai H0.
    parser.add_argument("--seq-length", type=int, default=7, help="Panjang sekuens input.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa file model jika sudah ada.",
    )
    # Hasil ini dikembalikan sebagai output fungsi `parse_args` untuk tahap berikutnya.
    return parser.parse_args()


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> int:
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parse_args()
    # `output_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"Model sudah ada: {output_path}")
        print("Gunakan --overwrite jika ingin menimpa.")
        # Hasil ini dikembalikan sebagai output fungsi `main` untuk tahap berikutnya.
        return 0

    model = build_model(img_size=args.img_size, seq_length=args.seq_length)
    model.save(output_path)
    print(f"Model bootstrap tersimpan: {output_path}")
    print("Catatan: model ini belum dilatih dan hanya untuk validasi integrasi runtime.")
    # Hasil ini dikembalikan sebagai output fungsi `main` untuk tahap berikutnya.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
