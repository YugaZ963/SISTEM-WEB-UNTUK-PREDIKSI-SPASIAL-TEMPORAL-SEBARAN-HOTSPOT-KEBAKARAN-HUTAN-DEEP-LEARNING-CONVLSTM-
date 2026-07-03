# Sistem Web Prediksi Hotspot (ConvLSTM Historical Risk Patch)

## Dokumentasi Terbaru yang Harus Dibaca

Dokumentasi utama terbaru ada di:

- `DOKUMENTASI_PROJECT_TERBARU_20260419.md`

File tersebut menjelaskan kondisi project paling baru setelah integrasi model
`historical_risk_patch_160` pada 2026-04-16, termasuk alur sistem, model aktif,
metrik evaluasi, cara menjalankan web, cara membaca hasil, dan batasan sistem.

Catatan:

- Beberapa dokumen lama masih berguna sebagai arsip pengembangan.
- Jika ada perbedaan informasi antara dokumen lama dan
  `DOKUMENTASI_PROJECT_TERBARU_20260419.md`, gunakan dokumen terbaru sebagai
  rujukan utama.

Implementasi web ini mengikuti dokumen:

- `Sistem Web Skripsi TA.md`
- `Sistem Web Skripsi TA Tahap-Tahap Pengembangan Sistem Web.md`
- `Sistem Web Skripsi TA Tahap-Tahap Implementasi Sistem Web.md`

Dokumen informasi terbaru project dan model ConvLSTM:

- `DOKUMENTASI_PROJECT_TERBARU_20260419.md`
- `backend/models/historical_risk_patch_20260416/README.md`

## Struktur Proyek

```text
backend/
  main.py
  routes/
    predict.py
  services/
    validator.py
    preprocess.py
    inference.py
    postprocess.py
    storage.py
  storage/
    inputs/
    outputs/
  templates/
    index.html
  static/
    app.js
    styles.css
```

## Fitur yang Sudah Diimplementasikan

1. Upload 7 file citra berurutan `H-6` s.d. `H0`.
2. Validasi jumlah, format, MIME type, ukuran, dan urutan nama file.
3. Penyimpanan input/output per `prediction_id` (UUID).
4. Preprocessing ke tensor `(1, 7, H, W, C)`.
5. Prediksi hotspot berbasis ConvLSTM historical risk patch sebagai default, dengan fallback heuristik temporal-spasial jika model gagal dimuat.
6. Postprocessing hasil:
   - `heatmap_H+n.png`
   - `overlay_H+n.png`
   - `binary_H+n.png` (jika threshold diisi)
   - `probability_H+n.npy`
7. Endpoint API:
   - `POST /api/predict`
   - `GET /api/predictions`
   - `GET /api/predictions/{prediction_id}`
   - `GET /api/predictions/{prediction_id}/download` (ZIP input+output)
   - `GET /api/runtime/status` (status engine/model runtime)
8. Frontend web:
   - halaman upload
   - halaman hasil
   - riwayat prediksi
   - tombol reset form
   - unduh ZIP semua hasil per prediksi

## Cara Menjalankan (Windows / PowerShell)

1. Aktifkan virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependency web:

```powershell
python -m pip install -r requirements.txt
```

3. (Opsional, untuk ConvLSTM) install dependency ML:

```powershell
python -m pip install -r requirements-ml.txt
```

4. Jalankan server:

```powershell
python -m uvicorn backend.main:app --reload
```

5. Buka:

```text
http://127.0.0.1:8000
```

## Catatan Engine Prediksi

- Default model profile:
  - `historical_risk_patch_160`
- File model default:
  - `backend/models/historical_risk_patch_20260416/best_model_convlstm_historical_risk_patch.keras`
- Mode inferensi default:
  - `patch_stitch`
- Input model:
  - 7 frame
  - patch `160x160`
  - 1 channel mask hotspot merah
- Preprocessing:
  - `hotspot_red_mask`
  - `input_dilation_kernel = 5`
- Threshold rekomendasi:
  - `0.55`
- Jika model/TensorFlow belum tersedia, sistem otomatis fallback ke
  `heuristic-temporal-v1` agar aplikasi tetap bisa berjalan.
- Cek status runtime aktif via endpoint:
  - `GET http://127.0.0.1:8000/api/runtime/status`

Untuk memakai model lama:

```powershell
$env:HOTSPOT_MODEL_PROFILE = "legacy_128"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Untuk kembali ke model baru:

```powershell
Remove-Item Env:\HOTSPOT_MODEL_PROFILE
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

## Bootstrap Model Integrasi (Arsip untuk Kondisi Lama)

Bagian ini hanya relevan jika model final belum ada. Pada kondisi terbaru,
model `historical_risk_patch_160` sudah disimpan di folder
`backend/models/historical_risk_patch_20260416/`.

Untuk mengaktifkan jalur runtime ConvLSTM tanpa menunggu training final, buat model bootstrap (arsitektur sama, bobot belum dilatih):

```powershell
.\.venv\Scripts\python.exe .\tools\bootstrap_convlstm_model.py --output Ipynb/best_model_convlstm.keras
```

Setelah itu restart server dan cek `/api/runtime/status`.

## Pengujian Otomatis

1. Install dependency dev:

```powershell
python -m pip install -r requirements-dev.txt
```

2. Jalankan test:

```powershell
python -m pytest -q
```

## Evaluasi Metrik Model (Saat Dataset Final Siap)

Gunakan script evaluasi mask prediksi vs ground truth:

```powershell
.\.venv\Scripts\python.exe .\tools\evaluate_convlstm_metrics.py `
  --pred-dir .\data\pred_masks `
  --gt-dir .\data\gt_masks `
  --threshold 0.5 `
  --output-json .\reports\metrics.json
```

Metrik yang dihitung: `precision`, `recall`, `f1_score`, `iou`, `accuracy`.

## Log Perubahan Project

Project ini memakai file log terpusat:

- `LOG_PERUBAHAN_PROJECT.md`

Setiap perubahan wajib ditambahkan sebagai entri baru (append), jangan menghapus riwayat lama.

Untuk menambahkan entri log otomatis, gunakan:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\tambah-log-perubahan.ps1 `
  -Judul "Judul Perubahan" `
  -Ringkasan "Ringkasan perubahan" `
  -Kategori "Fitur" `
  -Pelaksana "Nama Anda" `
  -LatarBelakang "Alasan perubahan" `
  -FileTerdampak "backend/main.py","backend/routes/predict.py" `
  -DetailTeknis "Perubahan A","Perubahan B" `
  -Dampak "Dampak 1","Dampak 2" `
  -Verifikasi "Uji endpoint berhasil" `
  -Risiko "Tidak ada risiko mayor" `
  -RencanaLanjut "Tambah test otomatis"
```

Catatan:

- Parameter `-Judul` dan `-Ringkasan` wajib diisi.
- Parameter lain opsional, tetapi disarankan tetap diisi agar log lengkap.

## Format Input Wajib

- Jumlah file: tepat 7 file.
- Nama file:
  - `H-6.png`
  - `H-5.png`
  - `H-4.png`
  - `H-3.png`
  - `H-2.png`
  - `H-1.png`
  - `H0.png`
- Ekstensi: `.png`, `.jpg`, `.jpeg`
- Ukuran maksimal: 5MB per file.
