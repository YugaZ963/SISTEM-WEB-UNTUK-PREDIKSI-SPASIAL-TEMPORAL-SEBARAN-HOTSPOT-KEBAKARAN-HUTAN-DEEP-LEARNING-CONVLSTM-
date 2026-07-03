# Historical Risk Patch Model 20260416

Folder ini menyimpan model ConvLSTM baru yang dipakai web sebagai default.

## Artifact

- `best_model_convlstm_historical_risk_patch.keras`
- `training_report_best_1epoch.json`
- `training_report_epoch3_comparison.json`

## Profil Web

- Profile: `historical_risk_patch_160`
- Input model: 7 frame, patch `160x160`, 1 channel mask hotspot merah
- Mode inference web: `patch_stitch`
- Input dilation kernel: `5`
- Threshold rekomendasi: `0.55`

Backend tidak lagi me-resize seluruh peta langsung ke `128x128` untuk model ini.
Web membuat mask hotspot dari setiap citra, memotongnya menjadi patch `160x160`,
memprediksi per patch, lalu menyatukan kembali hasilnya menjadi heatmap ukuran
gambar input.

## Ringkasan Metrik Training

Model default memakai run 1 epoch karena metrik F1/IoU sedikit lebih baik daripada
run 3 epoch.

- Standard test precision: `0.1601`
- Standard test recall: `0.2919`
- Standard test F1: `0.2068`
- Standard test IoU: `0.1153`
- Buffered test precision: `0.1949`
- Buffered test recall: `0.3474`
- Buffered test F1: `0.2497`
- Buffered test IoU: `0.1427`

Run 3 epoch tetap disimpan sebagai pembanding. Pada run 3 epoch, training
mengembalikan bobot terbaik dari epoch 1 sehingga tidak dijadikan default.

## Cara Memakai Model Lama

Jika perlu menjalankan web dengan model lama, set environment variable sebelum
menjalankan server:

```powershell
$env:HOTSPOT_MODEL_PROFILE = "legacy_128"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Untuk kembali ke model baru:

```powershell
Remove-Item Env:\HOTSPOT_MODEL_PROFILE
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```
