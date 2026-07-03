# Ringkasan Kondisi Project Terkini

Penjelasan kondisi project skripsi saat ini secara ringkas, detail, dan jelas. Isi dokumen ini mengacu pada kondisi kode, model, artefak training, dokumentasi, dan hasil pengujian yang ada di repository `sistem-web-skripsi-ta` per 24 April 2026.

Fokus utama penelitian ini adalah membangun sistem web yang dapat menerima 7 citra historis hotspot, memprosesnya menjadi input model, kemudian menghasilkan prediksi area risiko hotspot untuk hari berikutnya atau `H+1`.

Project ini tidak lagi memakai framing lama berupa prediksi grid MODIS `128x128` atau `256x256` sebagai implementasi utama. Kondisi terbaru project sudah beralih ke pendekatan **historical risk patch** dengan input citra hotspot format PNG berukuran asli, lalu diproses menggunakan patch `160x160`.

## 1. Ringkasan Singkat Kondisi Project

Secara umum, kondisi project saat ini dapat diringkas sebagai berikut:

1. Sistem web sudah terintegrasi dengan model ConvLSTM terbaru.
2. Model default yang dipakai web adalah `historical_risk_patch_160`.
3. Dataset utama yang digunakan untuk model terbaru terdiri dari 1060 citra hotspot historis format PNG.
4. Tahap training, evaluasi, integrasi web, dan pengujian alur API utama sudah selesai.
5. Model sudah dapat dipakai untuk menghasilkan prediksi indikatif area risiko hotspot, tetapi belum layak diklaim sebagai sistem operasional presisi tinggi.
6. Dokumentasi teknis project, laporan skripsi revisi, dan penjelasan BAB IV-BAB V juga sudah diperbarui agar sesuai dengan kondisi implementasi terbaru.

Dengan kata lain, project saat ini sudah berada pada tahap **prototype akademik yang berjalan end-to-end**, bukan lagi sekadar rancangan atau baseline awal.

## 2. Tujuan Sistem yang Sudah Berhasil Diimplementasikan

Sistem yang ada saat ini sudah memenuhi tujuan implementasi berikut:

1. Menerima 7 citra historis hotspot berurutan dari `H-6` sampai `H0`.
2. Memvalidasi jumlah file, nama file, format file, dan ukuran file.
3. Mengekstraksi hotspot merah dari citra sebagai representasi input model.
4. Menjalankan model ConvLSTM untuk prediksi area risiko hotspot.
5. Menghasilkan output visual berupa heatmap, overlay, binary mask, dan file probabilitas.
6. Menyimpan riwayat hasil prediksi beserta metadata prosesnya.
7. Menyediakan endpoint untuk memeriksa status runtime model yang sedang aktif.

## 3. Kondisi Dataset Terbaru

Dataset yang digunakan pada model terbaru memiliki karakteristik berikut:

| Komponen               | Kondisi terbaru                           |
| ---------------------- | ----------------------------------------- |
| Jenis data             | Citra hotspot historis                    |
| Format data            | PNG                                       |
| Jumlah frame           | 1060 citra                                |
| Rentang tanggal        | 2023-01-01 sampai 2025-11-25              |
| Ukuran sumber citra    | 1528 x 773 piksel                         |
| Panjang sequence input | 7 frame                                   |
| Target prediksi        | H+1                                       |
| Pembagian data         | 737 training, 158 validation, 158 testing |
| Cara split             | Kronologis berbasis waktu                 |

Setiap sampel dibentuk dari 7 citra historis, yaitu `H-6`, `H-5`, `H-4`, `H-3`, `H-2`, `H-1`, dan `H0`, untuk memprediksi area risiko hotspot pada `H+1`.

Salah satu karakteristik penting dataset ini adalah **class imbalance yang sangat ekstrem**. Nilai positive mass ratio pada training sekitar `0,00084170`, yang berarti piksel hotspot sangat sedikit dibandingkan piksel background. Kondisi ini menjadi alasan utama mengapa evaluasi model tidak cukup hanya menggunakan accuracy, tetapi harus menekankan precision, recall, F1-score, IoU, dan buffered evaluation.

## 4. Perubahan Penting dari Kondisi Lama ke Kondisi Terbaru

Project ini telah mengalami perubahan penting dibandingkan kondisi lama. Perubahan tersebut dapat diringkas sebagai berikut:

| Aspek              | Kondisi lama                                         | Kondisi terbaru                                                                                 |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Model default web  | Model lama `legacy_128` / jalur lama resize langsung | `historical_risk_patch_160`                                                                     |
| Pendekatan input   | Resize langsung ke ukuran kecil                      | Patch `160x160` full-resolution                                                                 |
| Fokus output       | Cenderung ke prediksi hotspot titik/peta kecil       | Prediksi area risiko hotspot                                                                    |
| Mode inferensi web | `direct_resize` pada model lama                      | `patch_stitch`                                                                                  |
| Folder backup      | Belum dipisahkan                                     | Sistem lama disimpan di `backend_legacy_before_historical_risk_20260416`                        |
| File model default | Jalur model lama                                     | `backend/models/historical_risk_patch_20260416/best_model_convlstm_historical_risk_patch.keras` |

Perubahan ini penting karena sistem web saat ini memang sudah disesuaikan dengan model baru, bukan sekadar menambahkan file model tanpa mengubah alur inferensi.

## 5. Kondisi Model Terbaru yang Menjadi Default Web

Model terbaru yang sekarang dipakai web sebagai default adalah:

- Profile: `historical_risk_patch_160`
- File model: `backend/models/historical_risk_patch_20260416/best_model_convlstm_historical_risk_patch.keras`
- Input: 7 frame historis
- Patch size: `160x160`
- Channel input: 1 channel mask hotspot merah
- Preprocessing: `hotspot_red_mask`
- Input dilation kernel: `5`
- Label dilation kernel: `9`
- Label blur radius: `2.0`
- Loss: `wbce_dice_context`
- Threshold rekomendasi: `0.55`
- Mode inferensi web: `patch_stitch`

Secara arsitektur, model menggunakan ConvLSTM untuk membaca pola spasial-temporal dari sequence citra hotspot, lalu menghasilkan peta probabilitas area risiko.

Model default web dipilih dari hasil **run 1 epoch**, bukan run 3 epoch, karena:

1. Hasil test standard F1-score dan IoU run 1 sedikit lebih baik.
2. Hasil buffered test run 1 juga sedikit lebih baik.
3. Pada run 3 epoch, training justru mengembalikan bobot terbaik dari epoch 1.

Dengan demikian, penggunaan run 1 epoch sebagai default dianggap lebih konsisten dan lebih sederhana untuk dipertanggungjawabkan.

## 6. Hasil Evaluasi Model Terbaru

Hasil evaluasi model terbaru yang saat ini dijadikan acuan adalah sebagai berikut.

### 6.1 Validation Best Threshold

- Threshold terbaik validation: `0.55`
- Validation precision: `0.1289`
- Validation recall: `0.1439`
- Validation F1-score: `0.1360`
- Validation IoU: `0.0730`
- Validation accuracy: `0.9852`

### 6.2 Test Standard

- Precision: `0.1601`
- Recall: `0.2919`
- F1-score: `0.2068`
- IoU: `0.1153`
- Accuracy: `0.9659`

### 6.3 Test Buffered Radius 5

- Precision: `0.1949`
- Recall: `0.3474`
- F1-score: `0.2497`
- IoU: `0.1427`
- Accuracy: `0.9348`

### 6.4 Interpretasi Hasil

Interpretasi yang aman dan sesuai kondisi project saat ini adalah:

1. Model sudah mampu menangkap sebagian pola area risiko hotspot.
2. Hasil buffered lebih baik daripada hasil standard, yang berarti banyak prediksi berada di sekitar hotspot aktual.
3. Model lebih tepat dibaca sebagai **prediksi area risiko**, bukan prediksi titik hotspot presisi.
4. Accuracy tinggi tidak dijadikan bukti utama karena background sangat dominan.
5. Model layak diposisikan sebagai **prototype akademik**, bukan sistem operasional final.

## 7. Kondisi Integrasi Sistem Web

Sistem web saat ini menggunakan:

| Komponen             | Kondisi                           |
| -------------------- | --------------------------------- |
| Backend              | FastAPI                           |
| Frontend             | HTML, CSS, JavaScript             |
| Entry point          | `backend/main.py`                 |
| Route utama prediksi | `backend/routes/predict.py`       |
| Validasi input       | `backend/services/validator.py`   |
| Preprocessing        | `backend/services/preprocess.py`  |
| Inference            | `backend/services/inference.py`   |
| Postprocessing       | `backend/services/postprocess.py` |
| Penyimpanan hasil    | `backend/storage/`                |

Alur kerja web saat ini adalah:

1. Pengguna mengunggah 7 citra berurutan.
2. Backend memvalidasi file.
3. Sistem mengekstraksi hotspot merah dari tiap citra.
4. Sequence input dibentuk sesuai model aktif.
5. Jika model aktif adalah `historical_risk_patch_160`, citra dipotong menjadi patch `160x160`.
6. Model memprediksi probabilitas risiko per patch.
7. Hasil patch digabung kembali menjadi satu peta probabilitas ukuran gambar input.
8. Sistem membuat heatmap, overlay, binary mask, dan menyimpan metadata.
9. Riwayat prediksi dapat diakses kembali dari web.

Poin penting:

- Web terbaru sudah **menyesuaikan model baru secara default**.
- Model lama masih disimpan dan masih dapat dijalankan jika diperlukan melalui environment variable `HOTSPOT_MODEL_PROFILE=legacy_128`.
- Project juga memiliki folder backup sistem lama yaitu `backend_legacy_before_historical_risk_20260416`.

## 8. Kondisi Runtime dan Fallback

Backend project memiliki mekanisme fallback agar aplikasi tetap bisa berjalan jika TensorFlow atau model gagal dimuat.

Artinya:

1. Jika model ConvLSTM terbaru berhasil dimuat, sistem akan memakai backend TensorFlow.
2. Jika TensorFlow tidak tersedia atau model gagal dibuka, sistem akan turun ke fallback heuristik `heuristic-temporal-v1`.

Karena itu, status server hidup tidak selalu berarti model ConvLSTM aktif. Untuk memastikan runtime yang sedang dipakai, project menyediakan endpoint:

`GET /api/runtime/status`

Pada kondisi normal yang diharapkan, endpoint tersebut harus menunjukkan profile model baru atau backend ConvLSTM, bukan fallback heuristik.

## 9. Hasil Pengujian Project Saat Ini

Pengujian otomatis untuk alur API utama dijalankan kembali pada 24 April 2026 dengan command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api_prediction_flow.py -q
```

Hasil pengujian:

```text
4 passed in 25.88s
```

Makna hasil ini:

1. Alur endpoint utama masih berjalan.
2. Validasi input utama masih berfungsi.
3. Penyimpanan artefak output masih berjalan.
4. Integrasi route prediksi utama tidak sedang rusak.

Hasil ini penting karena menunjukkan bahwa project bukan hanya selesai di level dokumen, tetapi juga masih lolos pengujian fungsional dasar.

## 10. Hal yang Sudah Selesai

Bagian-bagian berikut dapat dinyatakan **sudah selesai atau sudah tersedia dalam bentuk yang dapat diuji**:

1. Integrasi model ConvLSTM terbaru ke sistem web.
2. Penyesuaian backend web ke profile model baru.
3. Penyimpanan backup sistem web lama.
4. Dokumentasi teknis project terbaru.
5. Draft laporan skripsi revisi sesuai kondisi implementasi.
6. Pengujian otomatis API dasar.
7. Penyediaan artefak model dan report training terbaru.

## 11. Hal yang Masih Menjadi Keterbatasan

Walaupun project sudah berjalan, masih ada beberapa keterbatasan penting:

1. Model belum memakai variabel cuaca seperti suhu, curah hujan, kelembapan, dan angin.
2. Model belum memakai data lingkungan seperti gambut, topografi, land cover, atau aktivitas manusia.
3. Kelas hotspot masih sangat minoritas sehingga prediksi masih sulit diseimbangkan.
4. Evaluasi utama masih berfokus pada `H+1`.
5. Model belum dapat diklaim sebagai sistem operasional presisi tinggi.
6. Jika runtime gagal memuat model, sistem masih dapat fallback ke heuristik, sehingga verifikasi runtime tetap penting.

## 12. Posisi Ilmiah Project Saat Ini

Posisi akademik yang paling tepat untuk project ini adalah:

> Project telah berhasil membangun prototype sistem web prediksi area risiko hotspot berbasis ConvLSTM yang berjalan end-to-end, menggunakan 7 citra historis sebagai input, serta menghasilkan output visual dan riwayat prediksi. Model terbaru sudah terintegrasi ke sistem web dan telah dievaluasi menggunakan metrik yang sesuai untuk data imbalanced. Namun, model saat ini masih lebih tepat diklaim sebagai prototype akademik untuk prediksi indikatif area risiko hotspot, bukan sebagai sistem operasional presisi tinggi.

## 13. Rencana Pengembangan Selanjutnya

Langkah pengembangan yang paling logis berikutnya adalah:

1. Menambahkan variabel cuaca dan lingkungan.
2. Menguji loss function lain yang lebih kuat untuk class imbalance.
3. Mengevaluasi horizon prediksi `H+2` sampai `H+7` secara lebih khusus.
4. Menambahkan visualisasi evaluasi yang lebih lengkap.
5. Mengoptimalkan kecepatan inferensi web.
6. Melakukan validasi yang lebih dekat ke kondisi lapangan.

## 14. Kesimpulan Umum

project skripsi ini sudah berada pada kondisi yang cukup matang untuk dipresentasikan sebagai hasil implementasi yang nyata. Model terbaru sudah tersimpan, web sudah disesuaikan, backup sistem lama sudah tersedia, dokumentasi inti sudah diperbarui, draft laporan skripsi sudah direvisi, dan pengujian API dasar telah lulus.

Dengan demikian, kondisi project saat ini :

1. Project yang sudah berjalan secara teknis.
2. Project yang sudah memiliki model terbaru dan integrasi web.
3. Project yang sudah memiliki laporan revisi yang lebih sesuai dengan implementasi.
4. Project yang masih memerlukan penguatan kualitas model untuk tahap pengembangan lanjutan.
