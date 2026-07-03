# Dokumen Pengujian BAB IV

Tanggal uji: 2026-03-12  
Lingkungan: Windows (PowerShell), Python 3.10, TensorFlow 2.15.1

## 1) Pengujian Fungsional API

Skenario diuji melalui `pytest` di folder `tests/`.

| No | Skenario | Expected | Hasil |
|---|---|---|---|
| 1 | Prediksi valid 7 file -> detail -> unduh ZIP | API berhasil, metadata lengkap, ZIP berisi input+output | Lulus |
| 2 | Input file kurang dari 7 | HTTP 400 validasi jumlah file | Lulus |
| 3 | MIME file bukan image | HTTP 400 validasi MIME | Lulus |
| 4 | Cek status runtime model | Endpoint status mengembalikan engine/model/candidates | Lulus |

Ringkasan eksekusi:

```text
python -m pytest -q
....                                                                     [100%]
4 passed in 8.13s
```

## 2) Status Integrasi ConvLSTM Runtime

Endpoint validasi runtime:

```text
GET /api/runtime/status
```

Hasil saat diuji:

- `prediction_engine`: `convlstm-tensorflow (best_model_convlstm.keras)`
- `model_path`: `Ipynb/best_model_convlstm.keras`
- `preprocess_mode`: `hotspot_red_mask`
- `tensorflow_version`: `2.15.1`

Catatan penting:

- Model aktif saat ini adalah **bootstrap model** (arsitektur ConvLSTM, bobot belum training final).
- Model final skripsi tetap harus diganti dengan artefak training dataset final.

## 3) Artefak Uji

- File test otomatis: `tests/test_api_prediction_flow.py`
- Script bootstrap model: `tools/bootstrap_convlstm_model.py`
- Log perubahan implementasi: `LOG_PERUBAHAN_PROJECT.md`
