# File anotasi dari `Ipynb/Skripsi_TA_Final_3_fixed(1).py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: kode pendukung project skripsi.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Export notebook eksperimen awal skripsi untuk pelatihan dan evaluasi model ConvLSTM.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Catatan asli dari pembuat kode.
#!/usr/bin/env python
# Catatan asli dari pembuat kode.
# coding: utf-8

# Catatan asli dari pembuat kode.
# # Implementasi Skripsi: Prediksi Hotspot Kebakaran Hutan Riau dengan ConvLSTM
# Catatan asli dari pembuat kode.
# 
# Catatan asli dari pembuat kode.
# Notebook ini adalah implementasi final untuk Skripsi yang menggabungkan:
# Catatan asli dari pembuat kode.
# 1.  **Dataset Asli**: Menggunakan data historis hotspot Riau (citra satelit harian) dari 2020 hingga Januari 2025.
# Catatan asli dari pembuat kode.
# 2.  **Ekstraksi Fitur Hotspot**: Menggunakan thresholding warna merah untuk mendeteksi titik api asli dari peta.
# Catatan asli dari pembuat kode.
# 3.  **Model ConvLSTM**: Arsitektur Deep Learning Spasial-Temporal yang telah diperbaiki.
# Catatan asli dari pembuat kode.
# 4.  **Visualisasi Realistis**: Overlay hasil prediksi dan data asli pada peta Riau.
# Catatan asli dari pembuat kode.
# 
# Catatan asli dari pembuat kode.
# **Metodologi** mengacu pada dokumen *Laporan Skripsi Yuga.pdf* dan *KONSEP 3 (Manus)*.

# Catatan asli dari pembuat kode.
# ## 1. Setup & Konfigurasi

# Catatan asli dari pembuat kode.
# In[ ]:


# Mengambil alat bantu/library yang diperlukan oleh file ini.
import os
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import glob
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import matplotlib.pyplot as plt
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import cv2
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import tensorflow as tf
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras.models import Sequential
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, Input
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from google.colab import files



# Catatan asli dari pembuat kode.
# Konfigurasi
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
IMG_SIZE = 128  # Ukuran citra input model (resize dari asli)
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
SEQ_LENGTH = 7  # Panjang sekuens input (7 hari)
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
PRED_HORIZON = 1 # Prediksi 1 hari ke depan

# Menampilkan informasi ke terminal agar proses mudah dicek.
print("TensorFlow Version:", tf.__version__)

# Catatan asli dari pembuat kode.
# Mount Google Drive (wajib jika dataset di MyDrive)
# Mencoba menjalankan proses yang mungkin gagal.
try:
    # Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
    from google.colab import drive
    # Melanjutkan langkah kerja pada bagian kode ini.
    drive.mount('/content/drive')
# Menangani kesalahan agar program tidak langsung berhenti.
except Exception as e:
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("Warning: drive.mount gagal / bukan di Colab:", e)


# Catatan asli dari pembuat kode.
# ## 2. Persiapan Dataset Asli (Ekstraksi Hotspot)
# Catatan asli dari pembuat kode.
# Kita akan memuat dataset gambar harian dan mengekstraksi **titik merah** (hotspot) dari peta.
# Catatan asli dari pembuat kode.
# Tidak ada lagi data dummy/simulasi. Kode akan error jika dataset tidak ditemukan.

# Catatan asli dari pembuat kode.
# In[ ]:


# Catatan asli dari pembuat kode.
# Path ke dataset (sesuaikan jika di Google Drive)
# Catatan asli dari pembuat kode.
# Pastikan Drive sudah di-mount terlebih dahulu (lihat cell Setup).
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/Skripsi_Hotspot_ConvLSTM/data/dataset_riau/Dataset_History_Fire_Hotspot_In_Riau_Province/"

# Catatan asli dari pembuat kode.
# Validasi path
# Mengecek syarat sebelum melanjutkan proses.
if not os.path.exists(DATASET_PATH):
    # Menghentikan proses dan memberi pesan kesalahan yang jelas.
    raise FileNotFoundError(
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"Folder dataset tidak ditemukan: {DATASET_PATH}\n"
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Cek kembali path dan pastikan Google Drive sudah di-mount."
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

# Catatan asli dari pembuat kode.
# Fungsi untuk memuat dan preprocessing gambar
# Membuat langkah kerja bernama `load_images_from_folder`.
def load_images_from_folder(folder_path):
    # Catatan asli dari pembuat kode.
    # Cari semua file jpg (termasuk jika ada subfolder), urutkan berdasarkan nama file
    # Menyimpan nilai ke `image_files` untuk dipakai pada langkah berikutnya.
    image_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True))
    # Menyimpan nilai ke `images` untuk dipakai pada langkah berikutnya.
    images = []
    # Menyimpan nilai ke `valid_files` untuk dipakai pada langkah berikutnya.
    valid_files = []

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"Ditemukan {len(image_files)} file gambar dalam dataset.")
    # Mengecek syarat sebelum melanjutkan proses.
    if len(image_files) > 0:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("Contoh 3 file pertama:", [os.path.basename(p) for p in image_files[:3]])

    # Mengecek syarat sebelum melanjutkan proses.
    if len(image_files) == 0:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Dataset kosong atau path salah.\n"
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Pastikan ada file *.jpg di folder tersebut (atau di subfoldernya)."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Mengulang proses untuk setiap data dalam daftar.
    for filename in image_files:
        # Menyimpan nilai ke `img` untuk dipakai pada langkah berikutnya.
        img = cv2.imread(filename)
        # Mengecek syarat sebelum melanjutkan proses.
        if img is None:
            # Catatan asli dari pembuat kode.
            # Skip file yang gagal dibaca (mis. corrupt)
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue

        # Catatan asli dari pembuat kode.
        # 1) Resize ke ukuran standar model
        # Menyimpan nilai ke `img` untuk dipakai pada langkah berikutnya.
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Catatan asli dari pembuat kode.
        # 2) Ekstraksi warna merah (hotspot) via HSV
        # Mengubah warna RGB ke HSV agar warna merah hotspot lebih mudah dipisahkan.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Menyimpan nilai ke `lower_red1` untuk dipakai pada langkah berikutnya.
        lower_red1 = np.array([0, 70, 50])
        # Menyimpan nilai ke `upper_red1` untuk dipakai pada langkah berikutnya.
        upper_red1 = np.array([10, 255, 255])
        # Menyimpan nilai ke `lower_red2` untuk dipakai pada langkah berikutnya.
        lower_red2 = np.array([170, 70, 50])
        # Menyimpan nilai ke `upper_red2` untuk dipakai pada langkah berikutnya.
        upper_red2 = np.array([180, 255, 255])

        # Menyimpan nilai ke `mask1` untuk dipakai pada langkah berikutnya.
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        # Menyimpan nilai ke `mask2` untuk dipakai pada langkah berikutnya.
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        # Menggabungkan hasil deteksi merah menjadi mask hotspot.
        mask = mask1 + mask2

        # Catatan asli dari pembuat kode.
        # 3) Normalisasi 0-1 (mask 0 atau 255)
        # Menyimpan nilai ke `img_norm` untuk dipakai pada langkah berikutnya.
        img_norm = (mask / 255.0).astype(np.float32)

        # Melanjutkan langkah kerja pada bagian kode ini.
        images.append(img_norm)
        # Melanjutkan langkah kerja pada bagian kode ini.
        valid_files.append(filename)

    # Mengecek syarat sebelum melanjutkan proses.
    if len(images) == 0:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Tidak ada gambar yang berhasil dibaca oleh OpenCV.\n"
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Kemungkinan file corrupt / bukan JPG valid / permission masalah."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.stack(images, axis=0), valid_files

# Catatan asli dari pembuat kode.
# Load data (WAJIB sukses)
# Menyimpan nilai ke `data_images, filenames` untuk dipakai pada langkah berikutnya.
data_images, filenames = load_images_from_folder(DATASET_PATH)
# Menyimpan nilai ke `data_images` untuk dipakai pada langkah berikutnya.
data_images = data_images[..., np.newaxis]  # (N, H, W, 1)

# Menampilkan informasi ke terminal agar proses mudah dicek.
print("Final Data Shape:", data_images.shape)


# Catatan asli dari pembuat kode.
# ## 3. Pembentukan Sekuens & Splitting Data
# Catatan asli dari pembuat kode.
# Mengubah data runtun waktu menjadi format supervised learning:
# Catatan asli dari pembuat kode.
# - **Input (X)**: 7 hari berturut-turut (t-6 ... t)
# Catatan asli dari pembuat kode.
# - **Target (y)**: 1 hari berikutnya (t+1)
# Catatan asli dari pembuat kode.
# 
# Catatan asli dari pembuat kode.
# **Strategi Splitting** (Sesuai Laporan Skripsi):
# Catatan asli dari pembuat kode.
# - **Training**: Data Historis (2020-2022)
# Catatan asli dari pembuat kode.
# - **Testing**: Data Baru (Nov 2024 - Jan 2025)

# Catatan asli dari pembuat kode.
# In[ ]:


# Catatan asli dari pembuat kode.
# Pastikan data_images sudah dibuat di cell sebelumnya
# Mengecek syarat sebelum melanjutkan proses.
if 'data_images' not in globals():
    # Menghentikan proses dan memberi pesan kesalahan yang jelas.
    raise NameError("data_images belum ada. Jalankan cell 'Persiapan Dataset Asli' terlebih dahulu (load dataset).")

# Membuat langkah kerja bernama `create_sequences`.
def create_sequences(data, seq_length):
    # Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
    X = []
    # Menyimpan nilai ke `y` untuk dipakai pada langkah berikutnya.
    y = []
    # Mengulang proses untuk setiap data dalam daftar.
    for i in range(len(data) - seq_length):
        # Melanjutkan langkah kerja pada bagian kode ini.
        X.append(data[i:i+seq_length])
        # Melanjutkan langkah kerja pada bagian kode ini.
        y.append(data[i+seq_length])
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.array(X), np.array(y)

# Catatan asli dari pembuat kode.
# Buat sekuens dari seluruh data
# Menyimpan nilai ke `X, y` untuk dipakai pada langkah berikutnya.
X, y = create_sequences(data_images, SEQ_LENGTH)
# Menampilkan informasi ke terminal agar proses mudah dicek.
print(f"Input Sequences (X): {X.shape} -> (Samples, TimeSteps, H, W, C)")
# Menampilkan informasi ke terminal agar proses mudah dicek.
print(f"Target Labels (y): {y.shape} -> (Samples, H, W, C)")

# Catatan asli dari pembuat kode.
# Split Data secara Manual berdasarkan index (Asumsi data urut waktu)
# Catatan asli dari pembuat kode.
# Jika menggunakan data simulasi/campuran, kita split 80:20
# Catatan asli dari pembuat kode.
# Jika data asli sudah terurut tahunnya, sesuaikan index pemotongannya

# Menyimpan nilai ke `split_idx` untuk dipakai pada langkah berikutnya.
split_idx = int(len(X) * 0.8)
# Menyimpan nilai ke `X_train, X_test` untuk dipakai pada langkah berikutnya.
X_train, X_test = X[:split_idx], X[split_idx:]
# Menyimpan nilai ke `y_train, y_test` untuk dipakai pada langkah berikutnya.
y_train, y_test = y[:split_idx], y[split_idx:]

# Menampilkan informasi ke terminal agar proses mudah dicek.
print(f"Training Data (Historis): {X_train.shape}")
# Menampilkan informasi ke terminal agar proses mudah dicek.
print(f"Testing Data (Terbaru): {X_test.shape}")


# Catatan asli dari pembuat kode.
# ## 4. Arsitektur Model ConvLSTM (Fixed Version)
# Catatan asli dari pembuat kode.
# Menggunakan arsitektur yang telah diperbaiki untuk menghindari error dimensi pada TensorFlow terbaru.

# Catatan asli dari pembuat kode.
# In[ ]:


# Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
model = Sequential([
    # Catatan asli dari pembuat kode.
    # Layer 1: Input Layer Eksplisit
    # Menyimpan nilai ke `Input(shape` untuk dipakai pada langkah berikutnya.
    Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 1)),

    # Catatan asli dari pembuat kode.
    # Layer 2: ConvLSTM2D (Encoder)
    # Catatan asli dari pembuat kode.
    # Menangkap pola spasial-temporal dari urutan input
    # Menyimpan nilai ke `ConvLSTM2D(filters` untuk dipakai pada langkah berikutnya.
    ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True),
    # Melanjutkan langkah kerja pada bagian kode ini.
    BatchNormalization(),

    # Catatan asli dari pembuat kode.
    # Layer 3: ConvLSTM2D (Decoder)
    # Catatan asli dari pembuat kode.
    # Mengembalikan sequence menjadi representasi spasial tunggal (return_sequences=False)
    # Menyimpan nilai ke `ConvLSTM2D(filters` untuk dipakai pada langkah berikutnya.
    ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False),
    # Melanjutkan langkah kerja pada bagian kode ini.
    BatchNormalization(),

    # Catatan asli dari pembuat kode.
    # Layer 4: Output Layer (Conv2D)
    # Catatan asli dari pembuat kode.
    # Menghasilkan peta probabilitas hotspot (1 channel output)
    # Menyimpan nilai ke `Conv2D(filters` untuk dipakai pada langkah berikutnya.
    Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
# Menutup susunan data atau perintah yang dimulai sebelumnya.
])

# Menyimpan nilai ke `model.compile(optimizer` untuk dipakai pada langkah berikutnya.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Melanjutkan langkah kerja pada bagian kode ini.
model.summary()


# Catatan asli dari pembuat kode.
# ## 5. Training Model

# Catatan asli dari pembuat kode.
# In[ ]:


# Catatan asli dari pembuat kode.
# Callback untuk menyimpan model terbaik
# Menyimpan nilai ke `checkpoint` untuk dipakai pada langkah berikutnya.
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    # Melanjutkan langkah kerja pada bagian kode ini.
    'best_model_convlstm.keras',
    # Menyimpan nilai ke `monitor` untuk dipakai pada langkah berikutnya.
    monitor='val_loss',
    # Menyimpan nilai ke `save_best_only` untuk dipakai pada langkah berikutnya.
    save_best_only=True,
    # Menyimpan nilai ke `mode` untuk dipakai pada langkah berikutnya.
    mode='min',
    # Menyimpan nilai ke `verbose` untuk dipakai pada langkah berikutnya.
    verbose=1
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)

# Menyimpan nilai ke `history` untuk dipakai pada langkah berikutnya.
history = model.fit(
    # Melanjutkan langkah kerja pada bagian kode ini.
    X_train, y_train,
    # Menyimpan nilai ke `epochs` untuk dipakai pada langkah berikutnya.
    epochs=20,
    # Menyimpan nilai ke `batch_size` untuk dipakai pada langkah berikutnya.
    batch_size=4,
    # Menyimpan nilai ke `validation_data` untuk dipakai pada langkah berikutnya.
    validation_data=(X_test, y_test),
    # Menyimpan nilai ke `callbacks` untuk dipakai pada langkah berikutnya.
    callbacks=[checkpoint]
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)

# Catatan asli dari pembuat kode.
# Plot Loss
# Menyimpan nilai ke `plt.figure(figsize` untuk dipakai pada langkah berikutnya.
plt.figure(figsize=(10, 4))
# Menyimpan nilai ke `plt.plot(history.history['loss'], label` untuk dipakai pada langkah berikutnya.
plt.plot(history.history['loss'], label='Train Loss')
# Menyimpan nilai ke `plt.plot(history.history['val_loss'], label` untuk dipakai pada langkah berikutnya.
plt.plot(history.history['val_loss'], label='Val Loss')
# Melanjutkan langkah kerja pada bagian kode ini.
plt.title('Model Training Loss')
# Melanjutkan langkah kerja pada bagian kode ini.
plt.xlabel('Epoch')
# Melanjutkan langkah kerja pada bagian kode ini.
plt.ylabel('Loss')
# Melanjutkan langkah kerja pada bagian kode ini.
plt.legend()
# Melanjutkan langkah kerja pada bagian kode ini.
plt.show()


# Catatan asli dari pembuat kode.
# ## 6. Evaluasi & Visualisasi Overlay Peta (Realistis)
# Catatan asli dari pembuat kode.
# Bagian ini menampilkan hasil prediksi yang di-overlay ke peta asli Riau (`peta_riau.jpg`).
# Catatan asli dari pembuat kode.
# Semua gambar (Input, Ground Truth, Prediksi) akan memiliki background peta agar mudah dipahami.

# Catatan asli dari pembuat kode.
# In[ ]:


# Catatan asli dari pembuat kode.
# Upload Peta Background (Jika belum ada)
# Catatan asli dari pembuat kode.
# User harus upload 'peta_riau.jpg' ke Colab
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
MAP_FILENAME = 'peta_riau.jpg'

# Mengecek syarat sebelum melanjutkan proses.
if not os.path.exists(MAP_FILENAME):
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("Silakan upload file 'peta_riau.jpg'...")
    # Menyimpan nilai ke `uploaded` untuk dipakai pada langkah berikutnya.
    uploaded = files.upload()
    # Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
    MAP_FILENAME = next(iter(uploaded))

# Catatan asli dari pembuat kode.
# Load Peta Background
# Menyimpan nilai ke `base_map` untuk dipakai pada langkah berikutnya.
base_map = cv2.imread(MAP_FILENAME)
# Mengecek syarat sebelum melanjutkan proses.
if base_map is not None:
    # Catatan asli dari pembuat kode.
    # base_map = cv2.cvtColor(base_map, cv2.COLOR_BGR2RGB)
    # Menyimpan nilai ke `base_map` untuk dipakai pada langkah berikutnya.
    base_map = cv2.resize(base_map, (IMG_SIZE, IMG_SIZE))
# Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
else:
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("Warning: Peta background tidak ditemukan/gagal dimuat. Menggunakan background hitam.")
    # Menyimpan nilai ke `base_map` untuk dipakai pada langkah berikutnya.
    base_map = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# Catatan asli dari pembuat kode.
# Lakukan Prediksi pada Data Test Terakhir
# Menyimpan nilai ke `test_seq` untuk dipakai pada langkah berikutnya.
test_seq = X_test[-1:] # Ambil 1 sampel sequence terakhir
# Menyimpan nilai ke `pred_result` untuk dipakai pada langkah berikutnya.
pred_result = model.predict(test_seq)

# Catatan asli dari pembuat kode.
# --- FUNGSI BANTUAN UNTUK OVERLAY ---
# Membuat langkah kerja bernama `create_overlay`.
def create_overlay(mask_data, background_img, color_map=cv2.COLORMAP_JET, threshold=0.1):
    # Catatan asli dari pembuat kode.
    # 0) Validasi background
    # Mengecek syarat sebelum melanjutkan proses.
    if background_img is None:
        # Menyimpan nilai ke `background_img` untuk dipakai pada langkah berikutnya.
        background_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Catatan asli dari pembuat kode.
    # Pastikan background 3-channel uint8
    # Menyimpan nilai ke `bg` untuk dipakai pada langkah berikutnya.
    bg = background_img.copy()
    # Mengecek syarat sebelum melanjutkan proses.
    if bg.ndim == 2:
        # Menyimpan nilai ke `bg` untuk dipakai pada langkah berikutnya.
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    # Mengecek syarat sebelum melanjutkan proses.
    if bg.dtype != np.uint8:
        # Menyimpan nilai ke `bg` untuk dipakai pada langkah berikutnya.
        bg = np.clip(bg, 0, 255).astype(np.uint8)

    # Catatan asli dari pembuat kode.
    # 1) Mask -> 2D float
    # Menggabungkan hasil deteksi merah menjadi mask hotspot.
    mask = np.squeeze(mask_data).astype(np.float32)

    # Catatan asli dari pembuat kode.
    # 2) Threshold -> uint8 0/255
    # Menyimpan nilai ke `mask_thresh` untuk dipakai pada langkah berikutnya.
    mask_thresh = (mask > threshold).astype(np.uint8) * 255

    # Catatan asli dari pembuat kode.
    # 3) Kalau tidak ada hotspot, langsung return background
    # Menyimpan nilai ke `hotspot_area` untuk dipakai pada langkah berikutnya.
    hotspot_area = mask_thresh > 0
    # Mengecek syarat sebelum melanjutkan proses.
    if not np.any(hotspot_area):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return bg

    # Catatan asli dari pembuat kode.
    # 4) Heatmap dari mask
    # Menyimpan nilai ke `heatmap` untuk dipakai pada langkah berikutnya.
    heatmap = cv2.applyColorMap(mask_thresh, color_map)

    # Catatan asli dari pembuat kode.
    # 5) Blending hanya pada pixel hotspot
    # Menyimpan nilai ke `overlay` untuk dipakai pada langkah berikutnya.
    overlay = bg.copy()

    # Menyimpan nilai ke `bg_pixels` untuk dipakai pada langkah berikutnya.
    bg_pixels = bg[hotspot_area]
    # Menyimpan nilai ke `hm_pixels` untuk dipakai pada langkah berikutnya.
    hm_pixels = heatmap[hotspot_area]

    # Catatan asli dari pembuat kode.
    # Guard tambahan kalau ternyata kosong (double safety)
    # Mengecek syarat sebelum melanjutkan proses.
    if bg_pixels.size == 0 or hm_pixels.size == 0:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return overlay

    # Menyimpan nilai ke `blended` untuk dipakai pada langkah berikutnya.
    blended = cv2.addWeighted(bg_pixels, 0.6, hm_pixels, 0.4, 0)
    # Menyimpan nilai ke `overlay[hotspot_area]` untuk dipakai pada langkah berikutnya.
    overlay[hotspot_area] = blended

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return overlay


# Catatan asli dari pembuat kode.
# --- PROSES VISUALISASI ---

# Catatan asli dari pembuat kode.
# 1. Siapkan Gambar Input Terakhir (t)
# Menyimpan nilai ke `input_mask` untuk dipakai pada langkah berikutnya.
input_mask = test_seq[0, -1, :, :, 0] # Ambil frame terakhir
# Menyimpan nilai ke `vis_input` untuk dipakai pada langkah berikutnya.
vis_input = create_overlay(input_mask, base_map, threshold=0.1)

# Catatan asli dari pembuat kode.
# 2. Siapkan Gambar Ground Truth (t+1)
# Menyimpan nilai ke `gt_mask` untuk dipakai pada langkah berikutnya.
gt_mask = y_test[-1, :, :, 0]
# Menyimpan nilai ke `vis_gt` untuk dipakai pada langkah berikutnya.
vis_gt = create_overlay(gt_mask, base_map, threshold=0.1)

# Catatan asli dari pembuat kode.
# 3. Siapkan Gambar Prediksi (t+1)
# Menyimpan nilai ke `pred_mask` untuk dipakai pada langkah berikutnya.
pred_mask = pred_result[0, :, :, 0]
# Menyimpan nilai ke `vis_pred` untuk dipakai pada langkah berikutnya.
vis_pred = create_overlay(pred_mask, base_map, threshold=0.3) # Threshold prediksi bisa disesuaikan

# Catatan asli dari pembuat kode.
# --- TAMPILKAN ---
# Menyimpan nilai ke `plt.figure(figsize` untuk dipakai pada langkah berikutnya.
plt.figure(figsize=(18, 6))

# Melanjutkan langkah kerja pada bagian kode ini.
plt.subplot(1, 3, 1)
# Melanjutkan langkah kerja pada bagian kode ini.
plt.title("Input Terakhir (t)\n(Kondisi Kemarin)")
# Melanjutkan langkah kerja pada bagian kode ini.
plt.imshow(cv2.cvtColor(vis_input, cv2.COLOR_BGR2RGB))
# Melanjutkan langkah kerja pada bagian kode ini.
plt.axis('off')

# Melanjutkan langkah kerja pada bagian kode ini.
plt.subplot(1, 3, 2)
# Melanjutkan langkah kerja pada bagian kode ini.
plt.title("Ground Truth (t+1)\n(Kenyataan Hari Ini)")
# Melanjutkan langkah kerja pada bagian kode ini.
plt.imshow(cv2.cvtColor(vis_gt, cv2.COLOR_BGR2RGB))
# Melanjutkan langkah kerja pada bagian kode ini.
plt.axis('off')

# Melanjutkan langkah kerja pada bagian kode ini.
plt.subplot(1, 3, 3)
# Melanjutkan langkah kerja pada bagian kode ini.
plt.title("Prediksi Model (t+1)\n(Hasil Ramalan AI)")
# Melanjutkan langkah kerja pada bagian kode ini.
plt.imshow(cv2.cvtColor(vis_pred, cv2.COLOR_BGR2RGB))
# Melanjutkan langkah kerja pada bagian kode ini.
plt.axis('off')

# Melanjutkan langkah kerja pada bagian kode ini.
plt.show()

