Saya sudah buat script terpisah khusus untuk **Data Preparation**:

- [data_preparation_historical_risk_patch.py](C:/laragon/www/sistem-web-skripsi-ta/tools/data_preparation_historical_risk_patch.py)

Script ini sengaja dibuat:

- **tanpa training**
- **tanpa TensorFlow**
- hanya fokus ke:
  - ekstraksi `hotspot_red_mask`
  - pembuatan `risk_map`
  - sampling patch positif/negatif
  - ekstraksi patch `160x160`
  - ringkasan preprocessing yang enak di-SS

## Bagian kode yang bisa Anda SS dari file baru ini

1. Ekstraksi hotspot merah:

- [data_preparation_historical_risk_patch.py:93](C:/laragon/www/sistem-web-skripsi-ta/tools/data_preparation_historical_risk_patch.py:93)

2. Pembuatan `risk_map`:

- [data_preparation_historical_risk_patch.py:122](C:/laragon/www/sistem-web-skripsi-ta/tools/data_preparation_historical_risk_patch.py:122)

3. Sampling patch positif/negatif:

- [data_preparation_historical_risk_patch.py:153](C:/laragon/www/sistem-web-skripsi-ta/tools/data_preparation_historical_risk_patch.py:153)

4. Ekstraksi patch:

- [data_preparation_historical_risk_patch.py:186](C:/laragon/www\sistem-web-skripsi-ta/tools/data_preparation_historical_risk_patch.py:186)

5. Pembentukan patch entries:

- [data_preparation_historical_risk_patch.py:213](C:/laragon/www/sistem-web-skripsi-ta/tools/data_preparation_historical_risk_patch.py:213)

6. Ringkasan output terminal:

- [data_preparation_historical_risk_patch.py:620](C:/laragon/www/sistem-web-skripsi-ta/tools/data_preparation_historical_risk_patch.py:620)

## Command yang dipakai

Mode cepat untuk screenshot Data Preparation:

```powershell
python tools/data_preparation_historical_risk_patch.py --dataset-dir "Ipynb/Dataset History Fire Hotspot In Riau Province PNG" --image-extensions .png --save-preview-dir run/data_preparation_preview
```

Mode JSON:

```powershell
python tools/data_preparation_historical_risk_patch.py --dataset-dir "Ipynb/Dataset History Fire Hotspot In Riau Province PNG" --image-extensions .png --json
```

Mode statistik patch entries:

```powershell
python tools/data_preparation_historical_risk_patch.py --dataset-dir "Ipynb/Dataset History Fire Hotspot In Riau Province PNG" --image-extensions .png --with-patch-entry-stats --patch-entry-scope train
```

## Output yang sudah berhasil

Mode cepat sudah jalan dan menghasilkan ringkasan seperti:

- `Dataset valid: 1060`
- `Demo sample: input 2023-01-12 -> target 2023-01-13`
- `Input mask shape: (773, 1528)`
- `Risk map shape: (773, 1528)`
- `Input mask positive pixels: 234`
- `Risk map positive pixels (thresholded): 411`
- `Sampled patch centers: 5`
- `Example input patch shape: (160, 160, 1)`
- `Example target patch shape: (160, 160, 1)`

## Preview yang disimpan

Script juga sudah menyimpan file preview di:

- `run/data_preparation_preview/01_input_original.png`
- `run/data_preparation_preview/02_target_original.png`
- `run/data_preparation_preview/03_input_mask.png`
- `run/data_preparation_preview/04_risk_map.png`
- `run/data_preparation_preview/05_risk_map_with_patch_centers.png`
- `run/data_preparation_preview/06_example_input_patch.png`
- `run/data_preparation_preview/07_example_target_patch.png`

Ini justru sangat bagus untuk laporan, karena Anda bisa SS:

1. kode
2. output terminal
3. hasil visual preprocessing

## Saran screenshot yang paling kuat

Untuk **Data Preparation**, saya sarankan jangan kejar semuanya sekaligus.

Ambil:

1. **SS kode** `load_native_mask()` + `load_native_risk_map()`
2. **SS kode** `sample_patch_centers()` + `extract_patch()`
3. **SS terminal** hasil run mode cepat
4. **SS gambar preview**:
   - `03_input_mask.png`
   - `04_risk_map.png`
   - `05_risk_map_with_patch_centers.png`
   - `06_example_input_patch.png`
   - `07_example_target_patch.png`

## Catatan kritis

Bagian `patch entries train/val/test` exact memang **lebih berat**. Itu normal, karena tetap harus memindai banyak target map. Jadi untuk kebutuhan screenshot cepat:

- **jangan memaksakan output global exact**
- pakai **kode `build_patch_entries()` sebagai bukti implementasi**
- pakai **mode cepat + preview image** sebagai bukti proses

Itu lebih efisien dan lebih kuat secara presentasi.

Verifikasi:

- `python -m py_compile tools/data_preparation_historical_risk_patch.py` lolos
- mode cepat berhasil jalan
- preview preprocessing berhasil tersimpan

Kalau mau, saya bisa lanjut buat script terpisah yang sama untuk:

1. **Modeling**
2. **Evaluation**
3. **integrasi web `patch_stitch`**
