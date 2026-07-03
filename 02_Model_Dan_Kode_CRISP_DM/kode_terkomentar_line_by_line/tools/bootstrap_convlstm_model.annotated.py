# File anotasi dari `tools/bootstrap_convlstm_model.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Tool bootstrap model awal untuk memastikan file model dan struktur backend tersedia.

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
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import tensorflow as tf
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras import Sequential
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Input


# Membuat langkah kerja bernama `build_model`.
def build_model(img_size: int, seq_length: int) -> tf.keras.Model:
    # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
    model = Sequential(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            # Menyimpan nilai ke `Input(shape` untuk dipakai pada langkah berikutnya.
            Input(shape=(seq_length, img_size, img_size, 1)),
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            ConvLSTM2D(
                # Menyimpan nilai ke `filters` untuk dipakai pada langkah berikutnya.
                filters=32,
                # Menyimpan nilai ke `kernel_size` untuk dipakai pada langkah berikutnya.
                kernel_size=(3, 3),
                # Menyimpan nilai ke `padding` untuk dipakai pada langkah berikutnya.
                padding="same",
                # Menyimpan nilai ke `return_sequences` untuk dipakai pada langkah berikutnya.
                return_sequences=True,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            ),
            # Melanjutkan langkah kerja pada bagian kode ini.
            BatchNormalization(),
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            ConvLSTM2D(
                # Menyimpan nilai ke `filters` untuk dipakai pada langkah berikutnya.
                filters=32,
                # Menyimpan nilai ke `kernel_size` untuk dipakai pada langkah berikutnya.
                kernel_size=(3, 3),
                # Menyimpan nilai ke `padding` untuk dipakai pada langkah berikutnya.
                padding="same",
                # Menyimpan nilai ke `return_sequences` untuk dipakai pada langkah berikutnya.
                return_sequences=False,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            ),
            # Melanjutkan langkah kerja pada bagian kode ini.
            BatchNormalization(),
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


# Membuat langkah kerja bernama `parse_args`.
def parse_args() -> argparse.Namespace:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description="Buat model ConvLSTM bootstrap agar integrasi web bisa diuji tanpa menunggu model training final."
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--output",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="Ipynb/best_model_convlstm.keras",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Path output model .keras/.h5",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--img-size", type=int, default=128, help="Ukuran grid citra.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--seq-length", type=int, default=7, help="Panjang sekuens input.")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--overwrite",
        # Menyimpan nilai ke `action` untuk dipakai pada langkah berikutnya.
        action="store_true",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Timpa file model jika sudah ada.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser.parse_args()


# Membuat langkah kerja bernama `main`.
def main() -> int:
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parse_args()
    # Menyimpan nilai ke `output_path` untuk dipakai pada langkah berikutnya.
    output_path = Path(args.output).resolve()
    # Menyimpan nilai ke `output_path.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Mengecek syarat sebelum melanjutkan proses.
    if output_path.exists() and not args.overwrite:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"Model sudah ada: {output_path}")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("Gunakan --overwrite jika ingin menimpa.")
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return 0

    # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
    model = build_model(img_size=args.img_size, seq_length=args.seq_length)
    # Melanjutkan langkah kerja pada bagian kode ini.
    model.save(output_path)
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Model bootstrap tersimpan: {output_path}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("Catatan: model ini belum dilatih dan hanya untuk validasi integrasi runtime.")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return 0


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Menghentikan proses dan memberi pesan kesalahan yang jelas.
    raise SystemExit(main())
