# Folder Models

Backend sekarang memakai profil model. Profil default adalah:

- `historical_risk_patch_160`
- model: `backend/models/historical_risk_patch_20260416/best_model_convlstm_historical_risk_patch.keras`
- mode inference: `patch_stitch`
- input model: patch `160x160`
- preprocessing: ekstraksi mask hotspot merah dengan dilation kernel `5`
- threshold rekomendasi: `0.55`

Model lama tetap bisa dipakai dengan environment variable:

```powershell
$env:HOTSPOT_MODEL_PROFILE = "legacy_128"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Untuk model legacy, backend akan mencoba memuat model ConvLSTM dari salah satu path berikut:

- `backend/models/convlstm_model.h5`
- `backend/models/convlstm_saved_model/`
- `Ipynb/best_model_convlstm.keras`
- `Ipynb/best_model_convlstm.h5`

Selain path di atas, backend juga akan auto-discovery rekursif di folder
`Ipynb` untuk:

- file `*.keras`
- file `*.h5`
- folder SavedModel (`saved_model.pb`)

Jika file/folder model belum tersedia atau TensorFlow belum terpasang,
sistem otomatis fallback ke engine heuristik agar layanan tetap berjalan.

Catatan:
- Model historical-risk patch tidak memakai resize langsung seluruh gambar ke
  `128x128`. Backend memotong input menjadi patch `160x160`, menjalankan model
  per patch, lalu menggabungkan hasilnya menjadi heatmap ukuran gambar input.
- Jika model ConvLSTM berhasil dimuat, preprocessing input web memakai ekstraksi
  hotspot merah (`hotspot_red_mask`) sesuai pipeline training.
- Jika model final belum ada, Anda bisa membuat model bootstrap untuk tes
  integrasi dengan script `tools/bootstrap_convlstm_model.py`.
