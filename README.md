# Sistem Web Prediksi Spasial-Temporal Sebaran Hotspot Kebakaran Hutan Berbasis Deep Learning ConvLSTM

Repository ini berisi project sistem web untuk memprediksi area risiko hotspot kebakaran hutan di Provinsi Riau menggunakan pendekatan deep learning ConvLSTM. Sistem menerima tujuh citra historis berurutan dari H-6 sampai H0, memproses citra menjadi mask hotspot, menjalankan model ConvLSTM historical risk patch, lalu menghasilkan prediksi H+1 dalam bentuk heatmap, overlay peta, binary mask, riwayat prediksi, dan file unduhan hasil.

Project ini dikembangkan sebagai bagian dari skripsi/tugas akhir dengan fokus pada prediksi spasial-temporal sebaran hotspot berbasis citra sekuensial dan data pendukung NASA FIRMS.

## Ringkasan Project

- Wilayah studi: Provinsi Riau, Indonesia.
- Fokus prediksi: area risiko hotspot H+1 berdasarkan tujuh citra historis H-6 sampai H0.
- Model utama: ConvLSTM historical risk patch 160x160.
- Framework web: FastAPI, Jinja2, JavaScript, HTML, CSS.
- Output utama: heatmap probabilitas, overlay peta, binary mask, metadata prediksi, dan ZIP hasil.
- Data pendukung: citra hotspot, data NASA FIRMS pendamping, peta administrasi kabupaten/kota Riau, dan artefak evaluasi geospasial.

## Catatan Penting

Project ini tidak mengklaim mengambil data NASA FIRMS secara otomatis real-time. Pada implementasi web saat ini, sistem menerima input berupa citra historis yang diunggah pengguna. NASA FIRMS digunakan sebagai sumber data/rujukan hotspot dalam proses penelitian dan evaluasi, bukan sebagai proses sinkronisasi real-time otomatis di aplikasi web.

Prediksi yang dihasilkan adalah estimasi area risiko hotspot, bukan kepastian kejadian kebakaran. Hasil sistem perlu dipahami sebagai alat bantu analisis, bukan pengganti verifikasi lapangan atau keputusan resmi penanggulangan kebakaran.

## Fitur Utama

1. Upload tujuh citra historis hotspot berurutan dari H-6 sampai H0.
2. Validasi jumlah file, format nama file, ekstensi, MIME type, ukuran file, dan urutan input.
3. Preprocessing citra menjadi mask hotspot merah.
4. Inferensi model ConvLSTM berbasis patch 160x160 dengan mode patch stitching.
5. Fallback heuristik temporal-spasial jika model atau TensorFlow gagal dimuat.
6. Output visual berupa:
   - heatmap prediksi H+1,
   - overlay prediksi pada peta,
   - overlay dengan batas/nomor kabupaten/kota Riau,
   - binary mask berdasarkan threshold,
   - probability map dalam format `.npy`.
7. Riwayat prediksi berbasis `prediction_id`.
8. Unduh ZIP berisi input, output, metadata, dan hasil prediksi.
9. Endpoint API untuk prediksi, riwayat, detail prediksi, download, dan status runtime.
10. Kode training, evaluasi, validasi geospasial, dan dokumentasi pendukung untuk skripsi.

## Struktur Repository

```text
.
|-- 01_Project_Web_Demo/
|   |-- backend/
|   |   |-- main.py
|   |   |-- settings.py
|   |   |-- routes/
|   |   |-- services/
|   |   |-- static/
|   |   |-- templates/
|   |   |-- models/
|   |   `-- storage/
|   |-- data-admin/
|   |-- tests/
|   |-- requirements.txt
|   |-- requirements-ml.txt
|   `-- requirements-dev.txt
|
|-- 02_Model_Dan_Kode_CRISP_DM/
|   |-- tools/
|   |-- training_colab/
|   |-- script_training/
|   |-- patch_demo/
|   `-- kode_terkomentar_line_by_line/
|
|-- 03_Data_Pendukung/
|   |-- data-admin/
|   `-- data-firms-mentah-pendamping/
|
|-- 04_Hasil_Output_Dan_Evaluasi/
|   |-- artifacts/
|   |-- data_preparation_preview/
|   `-- dokumen_evaluasi/
|
|-- 05_Contoh_Input_Demo_7_Citra/
|   `-- contoh input H-6 sampai H0
|
|-- README_AUTO_VENV_VSCODE.md
|-- GITHUB_REPO_DESCRIPTION.md
`-- README.md
```

## Folder Penting

### `01_Project_Web_Demo`

Folder aplikasi web yang dapat dijalankan. Di dalamnya terdapat backend FastAPI, frontend dashboard, model ConvLSTM, konfigurasi, route API, service preprocessing/inference/postprocessing, serta test otomatis.

### `02_Model_Dan_Kode_CRISP_DM`

Folder berisi kode dan artefak pendukung tahapan CRISP-DM, termasuk script data understanding, data preparation, modeling, evaluation, deployment, training Colab, patch demo, dan versi kode terkomentar untuk lampiran/dokumentasi skripsi.

### `03_Data_Pendukung`

Folder berisi data pendukung seperti batas administrasi kabupaten/kota Riau dalam format GeoJSON dan data NASA FIRMS pendamping untuk kajian/evaluasi.

### `04_Hasil_Output_Dan_Evaluasi`

Folder berisi artefak hasil eksperimen, preview visual, validasi geospasial, rekap evaluasi, dan dokumen pengujian.

### `05_Contoh_Input_Demo_7_Citra`

Folder berisi contoh input demo tujuh citra historis untuk menguji sistem web.

## Arsitektur Alur Sistem

```text
Input 7 citra H-6 sampai H0
        |
        v
Validasi file dan urutan input
        |
        v
Preprocessing citra menjadi mask hotspot
        |
        v
Pemotongan citra menjadi patch 160x160
        |
        v
Inferensi ConvLSTM historical risk patch
        |
        v
Stitching hasil patch menjadi peta prediksi penuh
        |
        v
Postprocessing heatmap, overlay, binary mask, dan metadata
        |
        v
Tampilan dashboard web dan riwayat prediksi
```

## Model Prediksi

Model default yang digunakan aplikasi web:

- Profile: `historical_risk_patch_160`
- File model: `01_Project_Web_Demo/backend/models/historical_risk_patch_20260416/best_model_convlstm_historical_risk_patch.keras`
- Arsitektur utama: ConvLSTM
- Input: 7 frame historis
- Ukuran patch: 160x160
- Channel input: 1 channel mask hotspot merah
- Mode preprocessing: `hotspot_red_mask`
- Mode inferensi: `patch_stitch`
- Threshold rekomendasi: 0.55

Ringkasan metrik training model default berdasarkan artefak project:

| Metrik | Standard Test | Buffered Test |
|---|---:|---:|
| Precision | 0.1601 | 0.1949 |
| Recall | 0.2919 | 0.3474 |
| F1 Score | 0.2068 | 0.2497 |
| IoU | 0.1153 | 0.1427 |

Metrik tersebut menunjukkan bahwa project ini masih merupakan prototype penelitian. Nilai recall, precision, F1, dan IoU perlu terus ditingkatkan melalui pengayaan dataset, penambahan variabel lingkungan, pembandingan model, dan evaluasi yang lebih luas.

## Format Input

Sistem membutuhkan tepat tujuh file citra dengan urutan historis sebagai berikut:

```text
H-6.png
H-5.png
H-4.png
H-3.png
H-2.png
H-1.png
H0.png
```

Format bertanggal juga didukung, misalnya:

```text
H-6_2025-07-13.png
H-5_2025-07-14.png
H-4_2025-07-15.png
H-3_2025-07-16.png
H-2_2025-07-17.png
H-1_2025-07-18.png
H0_2025-07-19.png
```

Ketentuan input:

- Jumlah file: 7 file.
- Ekstensi: `.png`, `.jpg`, atau `.jpeg`.
- Ukuran maksimal: 5 MB per file.
- Urutan nama file harus merepresentasikan H-6 sampai H0.
- Jika nama file memakai tanggal, sistem dapat menghitung target H+1 dari tanggal H0.

## Output Sistem

Setiap prediksi disimpan berdasarkan `prediction_id` dan dapat menghasilkan file berikut:

```text
heatmap_H+1.png
overlay_H+1.png
overlay_no_numbers_H+1.png
overlay_plain_H+1.png
binary_H+1.png
probability_H+1.npy
params.json
prediction_outputs.zip
```

Keterangan output:

- `heatmap_H+1.png`: visualisasi probabilitas risiko hotspot.
- `overlay_H+1.png`: hasil prediksi yang ditumpangkan pada peta/citra referensi.
- `overlay_no_numbers_H+1.png`: overlay batas wilayah tanpa nomor kabupaten/kota.
- `overlay_plain_H+1.png`: overlay bersih tanpa batas administrasi.
- `binary_H+1.png`: binary mask berdasarkan threshold.
- `probability_H+1.npy`: array probabilitas hasil prediksi.
- `params.json`: metadata prediksi.
- `prediction_outputs.zip`: paket unduhan input dan output.

## Endpoint API

| Method | Endpoint | Fungsi |
|---|---|---|
| `POST` | `/api/predict` | Mengunggah input dan menjalankan prediksi H+1 |
| `GET` | `/api/predictions` | Mengambil daftar riwayat prediksi |
| `GET` | `/api/predictions/{prediction_id}` | Mengambil detail satu prediksi |
| `GET` | `/api/predictions/{prediction_id}/download` | Mengunduh ZIP input dan output prediksi |
| `GET` | `/api/runtime/status` | Mengecek status model, backend inferensi, dan konfigurasi runtime |

Dokumentasi API otomatis FastAPI dapat dibuka ketika server berjalan:

```text
http://127.0.0.1:8000/docs
```

## Instalasi dan Cara Menjalankan

### 1. Clone repository

```powershell
git clone https://github.com/YugaZ963/SISTEM-WEB-UNTUK-PREDIKSI-SPASIAL-TEMPORAL-SEBARAN-HOTSPOT-KEBAKARAN-HUTAN-DEEP-LEARNING-CONVLSTM-.git
cd SISTEM-WEB-UNTUK-PREDIKSI-SPASIAL-TEMPORAL-SEBARAN-HOTSPOT-KEBAKARAN-HUTAN-DEEP-LEARNING-CONVLSTM-
```

### 2. Masuk ke folder web demo

```powershell
cd 01_Project_Web_Demo
```

### 3. Buat dan aktifkan virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell menolak aktivasi venv karena execution policy, jalankan:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Lalu aktifkan kembali venv.

### 4. Install dependency utama

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Install dependency machine learning

```powershell
python -m pip install -r requirements-ml.txt
```

Dependency machine learning berisi TensorFlow untuk memuat dan menjalankan model ConvLSTM. Jika TensorFlow tidak tersedia atau gagal dimuat, aplikasi masih dapat berjalan dengan fallback heuristik, tetapi hasilnya bukan dari model ConvLSTM utama.

### 6. Jalankan server

```powershell
python -m uvicorn backend.main:app --reload
```

Buka browser:

```text
http://127.0.0.1:8000
```

## Cara Menjalankan dengan VS Code Lokal

Folder repository ini juga menyertakan konfigurasi `.vscode/settings.json` untuk lingkungan lokal pengembang. Pada komputer pengembang asli, terminal VS Code diarahkan ke folder:

```text
01_Project_Web_Demo
```

Dan memakai virtual environment dari project utama lokal. Jika repository dipindahkan ke komputer lain, sesuaikan path interpreter Python di `.vscode/settings.json` atau buat venv baru di dalam `01_Project_Web_Demo` seperti langkah instalasi di atas.

## Pengujian Otomatis

Install dependency development:

```powershell
python -m pip install -r requirements-dev.txt
```

Jalankan test:

```powershell
python -m pytest -q
```

Test otomatis memeriksa alur API prediksi, validasi input, dan respons endpoint utama.

## Teknologi yang Digunakan

- Python
- FastAPI
- Uvicorn
- TensorFlow/Keras
- NumPy
- Pillow
- Jinja2
- HTML, CSS, JavaScript
- Bootstrap
- GeoJSON
- Pytest

## Data dan Referensi Utama

- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- NASA FIRMS API: https://firms.modaps.eosdis.nasa.gov/api/
- FastAPI: https://fastapi.tiangolo.com/
- TensorFlow: https://www.tensorflow.org/

## Batasan Project

1. Sistem belum mengambil data NASA FIRMS secara otomatis real-time.
2. Input web masih berupa citra historis yang diunggah pengguna.
3. Model utama masih berbasis riwayat visual hotspot, belum menggabungkan variabel cuaca, kekeringan, vegetasi, gambut, land cover, topografi, dan faktor manusia secara penuh.
4. Hasil prediksi harus dibaca sebagai peta risiko, bukan kepastian kebakaran.
5. Metrik model masih perlu ditingkatkan dengan dataset yang lebih kaya dan evaluasi yang lebih luas.
6. Sistem ini merupakan prototype akademik untuk kebutuhan penelitian/skripsi.

## Rencana Pengembangan

Beberapa pengembangan yang disarankan:

1. Menggunakan FIRMS/VIIRS sebagai label H+1 dan membangun dataset multivariabel berbasis grid harian Riau.
2. Menambahkan variabel cuaca dan kekeringan seperti suhu, curah hujan, kelembapan, angin, soil moisture, dan Fire Weather Index.
3. Menambahkan variabel vegetasi/bahan bakar seperti NDVI, EVI, NDWI/NDMI, Land Surface Temperature, dan fuel moisture.
4. Menambahkan variabel lokal Riau seperti gambut, kedalaman gambut, kanal/drainase, jarak ke sungai, jarak ke jalan, perkebunan, HTI/sawit, dan permukiman.
5. Membandingkan ConvLSTM dengan XGBoost, LightGBM, Random Forest, U-Net temporal, ConvLSTM U-Net, atau model hybrid berbasis indeks bahaya kebakaran.
6. Memperluas horizon prediksi menjadi H+3, H+7, atau prediksi risiko bulanan.
7. Menambahkan evaluasi PR-AUC, ROC-AUC, F1, IoU, Brier Score, calibration curve, dan validasi geospasial per kabupaten/kota.

## Status Project

Project ini sudah memiliki aplikasi web demo, model ConvLSTM default, endpoint API, frontend dashboard, contoh input, data pendukung, dan artefak evaluasi. Project masih terbuka untuk pengembangan model, pengayaan variabel, peningkatan metrik, dan otomasi pipeline data.

## Author

Yuga Azka Al Razzak

## Lisensi

Lisensi repository belum ditentukan. Jika project ini digunakan sebagai referensi akademik, cantumkan sumber repository dan penulis project.

