# Laporan Riwayat Pelatihan Model ConvLSTM

Tanggal laporan: `2026-04-09 15:47:45 WIB`

Project:

- `C:\laragon\www\sistem-web-skripsi-ta`

Tujuan dokumen ini:

- menjelaskan **apa saja yang sudah Anda lakukan** dalam proses melatih model
- menjelaskan **percobaan yang berhasil**
- menjelaskan **percobaan yang gagal**
- menjelaskan **kenapa kualitas model masih rendah**
- menjelaskan **kenapa beberapa perubahan kode membantu dan beberapa perubahan lain justru tidak membantu**

Dokumen ini ditulis dengan bahasa yang lebih mudah dipahami untuk pemula IT.

Kalau ada istilah teknis, artinya saya jelaskan langsung dalam tanda kurung.

---

## 1. Ringkasan Singkat untuk Dipahami Dulu

Kalau seluruh perjalanan training model ini diringkas secara sederhana, hasilnya seperti ini:

1. Anda **sudah berhasil membuat model ConvLSTM berjalan**.
2. Anda **sudah berhasil menambah jumlah dataset** sampai sekitar `1060` gambar.
3. Anda **sudah mencoba banyak perbaikan kode dan konfigurasi training**.
4. Beberapa eksperimen memang **meningkatkan kualitas model**.
5. Tetapi peningkatannya **masih kecil**, belum sampai kategori model yang benar-benar kuat.
6. Penyebab utamanya bukan sekadar kode model, tetapi juga:
   - data yang dipakai masih terlalu sederhana
   - target hotspot terlalu kecil
   - gambar di-resize (diperkecil) sehingga titik hotspot mudah hilang
   - model hanya belajar dari gambar render (gambar visual) hotspot, bukan dari data geospasial mentah yang lebih kaya

Kesimpulan paling jujur:

- **Anda sudah berhasil membangun baseline model** (model dasar yang benar-benar jalan)
- **Anda belum berhasil mendapatkan model yang metriknya tinggi**
- tetapi **perjalanan eksperimen Anda valid dan bernilai untuk skripsi**

---

## 2. Apa Sebenarnya yang Sedang Dilatih?

Model yang Anda latih adalah **ConvLSTM**.

### Apa itu ConvLSTM?

ConvLSTM (Convolutional Long Short-Term Memory) adalah model deep learning (model pembelajaran mesin berbasis jaringan saraf) yang cocok untuk:

- data gambar
- data berurutan
- data yang punya hubungan waktu

Contohnya:

- video
- radar hujan
- peta hotspot harian

Dalam project Anda:

- input model = `7` gambar berurutan
- model mencoba memprediksi kondisi hotspot berikutnya

Jadi model Anda belajar pola seperti:

> “Kalau 7 hari terakhir hotspot muncul seperti ini, kemungkinan hotspot berikutnya muncul di mana?”

---

## 3. Data yang Dipakai Model

### 3.1 Dataset awal

Pada awal project, model dilatih memakai dataset gambar peta hotspot harian.

Isi gambar:

- peta wilayah
- titik hotspot berwarna merah

Yang dibaca model sebenarnya bukan gambar peta secara penuh, melainkan:

- sistem mencari warna merah
- warna merah diubah menjadi mask (mask = peta biner/angka yang menandai area hotspot)

### 3.2 Bentuk data training

Awalnya model dilatih dengan logika seperti ini:

- ambil gambar `H-6`, `H-5`, `H-4`, `H-3`, `H-2`, `H-1`, `H0`
- ubah menjadi tensor (tensor = bentuk data numerik multidimensi untuk model)
- model memprediksi `H+1`

### 3.3 Perubahan jumlah dataset

Dari riwayat percobaan yang tercatat:

- awalnya ada run dengan dataset sekitar `329` frame
- kemudian Anda menambah dataset menjadi sekitar `1059`
- lalu stabil di sekitar `1060` frame

Ini adalah langkah yang benar, karena:

- model butuh lebih banyak variasi data
- data yang terlalu sedikit membuat model sulit belajar

### 3.4 Masalah utama pada dataset

Walaupun jumlah gambar bertambah, masalah besarnya tetap ada:

1. titik hotspot merah sangat kecil
2. gambar disimpan sebagai `JPG`
3. gambar kemudian di-resize menjadi grid kecil seperti `128x128`
4. titik merah kecil jadi mudah hilang atau blur

Jadi masalah Anda bukan hanya “jumlah data kurang”.

Masalah yang lebih serius justru:

- **informasi hotspot di dalam gambar terlalu tipis**

---

## 4. Bentuk Perjalanan Training yang Sudah Anda Lakukan

Secara besar, perjalanan training model Anda bisa dibagi menjadi beberapa fase:

1. fase awal: model mulai bisa jalan di Colab
2. fase penambahan dataset
3. fase tuning 128x128
4. fase percobaan 256x256
5. fase loss function dan threshold tuning
6. fase improved search / eksperimen lanjutan
7. fase native patch `1528x773`
8. fase persiapan training lokal GPU / WSL2

Di bawah ini saya jelaskan satu per satu.

---

## 5. Fase Awal: Model Mulai Bisa Jalan

Pada fase awal, fokus utama bukan kualitas model, tetapi:

- membuat notebook Colab bisa berjalan
- membuat dataset terbaca
- membuat training menghasilkan model

### Yang berhasil di fase ini

1. Script training diubah menjadi notebook Colab
   - file:
     - `Ipynb/TRAIN_CONVLSTM_FINAL_GOOGLE_COLAB.py`
     - `Ipynb/TRAIN_CONVLSTM_FINAL_GOOGLE_COLAB.ipynb`

2. Error `argparse` (argparse = pembaca argumen command line) berhasil diperbaiki
   - supaya notebook tidak error oleh argumen bawaan Jupyter/Colab

3. Path dataset di Google Drive berhasil dibuat lebih fleksibel
   - supaya notebook tidak gagal hanya karena nama folder sedikit berbeda

### Arti fase ini

Ini fase yang penting, karena tanpa fase ini:

- model tidak akan pernah sampai ke tahap evaluasi

Jadi walaupun kualitas model belum bagus, fase awal ini tetap sebuah **keberhasilan teknis**.

---

## 6. Fase Penambahan Dataset: Dari Sangat Sedikit Menjadi Lebih Banyak

Setelah notebook berhasil jalan, Anda menambah jumlah data.

### Run yang sangat lemah

Ada run lama dengan dataset kecil:

- file:
  - `Ipynb/artifacts-2/convlstm_training_report_run_20260323_120141.json`

Metriknya:

- Precision = `0.0`
- Recall = `0.0`
- F1-score = `0.0`
- IoU = `0.0`

Artinya:

- model gagal total
- model tidak mendeteksi hotspot sama sekali

### Setelah dataset ditambah

Run berikutnya:

- `Ipynb/artifacts/convlstm_training_report_run_20260323_123547.json`

Metrik:

- Precision = `0.222222`
- Recall = `0.012793`
- F1-score = `0.024194`
- IoU = `0.012245`

Interpretasi:

- ini **lebih baik** daripada nol total
- model mulai mendeteksi hotspot
- tetapi masih sangat lemah

### Kenapa ini tetap penting?

Karena ini menunjukkan:

- menambah dataset memang membantu
- model mulai belajar sesuatu

Tetapi:

- dataset yang lebih banyak saja tidak cukup

---

## 7. Fase Perbaikan Baseline 128x128

Ini fase paling penting dalam perjalanan model Anda.

Di fase ini dilakukan beberapa perbaikan seperti:

- perbaikan ekstraksi hotspot
- menjaga hotspot kecil agar tidak langsung hilang
- weighted BCE + dice loss
- threshold sweep

### Apa itu weighted BCE?

Weighted BCE (weighted binary cross entropy) adalah loss function (fungsi kerugian / ukuran seberapa salah model) yang memberi bobot lebih besar pada kelas positif.

Ini penting karena:

- jumlah hotspot jauh lebih sedikit daripada background

### Apa itu dice loss?

Dice loss adalah loss yang cocok untuk segmentasi (segmentasi = memprediksi area/piksel, bukan hanya satu angka kelas).

Dice loss membantu model lebih fokus ke tumpang tindih antara prediksi dan target.

### Hasil run yang menjadi baseline penting

Run:

- `Ipynb/artifacts/convlstm_training_report_run_20260324_041641.json`

Metrik:

- Precision = `0.115265`
- Recall = `0.161505`
- F1-score = `0.134522`
- IoU = `0.072111`

### Mengapa run ini penting?

Karena pada saat itu:

- ini adalah lompatan besar dibanding run yang sebelumnya hampir nol
- model mulai masuk kategori “tidak jelek”

Artinya:

- model belum bagus
- tetapi sudah tidak gagal total

### Ini termasuk keberhasilan atau kegagalan?

Ini **keberhasilan parsial**.

Kenapa?

- berhasil menaikkan kualitas model secara nyata
- tetapi belum mencapai kualitas tinggi

---

## 8. Fase Percobaan 256x256

Setelah baseline 128 mulai lumayan, Anda mencoba menaikkan resolusi training menjadi `256x256`.

Logikanya masuk akal:

- kalau hotspot kecil, mungkin resolusi lebih tinggi akan membantu

### Run 256 yang tercatat

- `Ipynb/artifacts/convlstm_training_report_run_20260324_044821.json`

Metrik:

- Precision = `0.111759`
- Recall = `0.157193`
- F1-score = `0.130639`
- IoU = `0.069884`

### Dibanding baseline 128

Baseline `041641`:

- F1 = `0.134522`
- IoU = `0.072111`

Run 256:

- F1 = `0.130639`
- IoU = `0.069884`

### Kesimpulannya

- `256x256` **tidak mengalahkan** baseline `128x128`

### Kenapa bisa begitu?

Kemungkinan besar karena:

1. data dasarnya masih lemah
2. resolusi lebih tinggi membawa detail, tetapi juga membawa noise (noise = gangguan / informasi tidak penting)
3. model belum cukup kuat memanfaatkan detail tambahan

### Ini pelajaran penting

Anda belajar bahwa:

- **resolusi lebih tinggi tidak otomatis berarti hasil lebih baik**

Ini sangat penting untuk skripsi, karena menunjukkan Anda sudah melakukan eksperimen yang masuk akal, bukan asal memilih angka.

---

## 9. Fase Precision Naik, Tapi Recall Turun

Setelah beberapa tuning lagi, muncul run:

- `Ipynb/artifacts/convlstm_training_report_run_20260324_052358.json`

Metrik:

- Precision = `0.1531`
- Recall = `0.118063`
- F1-score = `0.133318`
- IoU = `0.071420`

### Arti hasil ini

Precision naik:

- model jadi lebih hati-hati
- lebih sedikit false positive (false positive = prediksi hotspot padahal sebenarnya bukan hotspot)

Tetapi recall turun:

- model jadi lebih sering melewatkan hotspot

### Kenapa ini tidak dianggap menang?

Karena untuk masalah hotspot:

- Anda tidak hanya butuh precision
- Anda juga butuh recall, F1, dan IoU

Kalau precision naik tapi recall turun terlalu banyak:

- model tetap belum lebih baik secara keseluruhan

### Pelajaran dari fase ini

Anda belajar bahwa:

- meningkatkan satu metrik saja belum tentu meningkatkan kualitas model secara umum

---

## 10. Fase Search Baseline dan Eksperimen Stabil

Di fase berikutnya, kode training diperbaiki agar bisa mencoba beberapa kandidat model yang masih dekat dengan baseline terbaik.

Tujuannya:

- tidak menebak-nebak lagi satu per satu secara manual
- tetapi membuat pencarian kandidat lebih sistematis

### Hasil-hasil baseline yang tersimpan

Di `Ipynb/artifacts`, ada beberapa model penting:

#### `baseline_dice_gt003_k3`

- Precision = `0.127063`
- Recall = `0.166123`
- F1-score = `0.143991`
- IoU = `0.077581`

#### `baseline_dice_gt003_k5`

- Precision = `0.073702`
- Recall = `0.251906`
- F1-score = `0.114039`
- IoU = `0.060467`

#### `baseline_dice_gt005_k3`

- Precision = `0.126896`
- Recall = `0.166909`
- F1-score = `0.144178`
- IoU = `0.077690`

#### `baseline_dice_gt005_k5`

- Precision = `0.139688`
- Recall = `0.161402`
- F1-score = `0.149762`
- IoU = `0.080942`

#### `dice_gt005_k3_aug`

- Precision = `0.12603`
- Recall = `0.165246`
- F1-score = `0.142998`
- IoU = `0.077005`

### Mana yang terbaik?

Dalam kumpulan baseline itu, yang paling seimbang adalah:

- `baseline_dice_gt005_k5`

File:

- `Ipynb/artifacts/best_model_convlstm_final_baseline_dice_gt005_k5.keras`
- `Ipynb/artifacts/convlstm_training_report_baseline_dice_gt005_k5.json`

### Kenapa ini dianggap terbaik?

Karena:

- F1 paling tinggi di kelompok baseline
- IoU paling tinggi di kelompok baseline
- precision dan recall masih cukup seimbang

### Apa pelajaran dari fase ini?

Anda belajar bahwa:

- loss `wbce_dice` lebih stabil dibanding beberapa eksperimen lain
- augmentasi (augmentasi = penambahan variasi data buatan seperti flip/rotasi) tidak memberi lompatan besar
- oversampling agresif (mengulang data positif terlalu sering) juga tidak otomatis membantu

---

## 11. Fase Improved Search dan Model Context

Setelah baseline stabil, Anda mencoba beberapa peningkatan baru.

Fitur yang ditambahkan saat itu antara lain:

- `wbce_dice_context`
- `feature_stack`
- `val_model_score`
- `PR-AUC`

### Apa itu context?

Context (konteks) artinya model tidak hanya melihat hotspot inti, tetapi juga area sekitar hotspot.

Harapannya:

- model bisa memahami “lingkungan sekitar hotspot”

### Hasil improved yang cukup positif

Di `Ipynb/artifacts-2`, ada model:

- `convlstm_training_report.json`
- model:
  - `best_model_convlstm_final.keras`

Metriknya:

- Precision = `0.151159`
- Recall = `0.163883`
- F1-score = `0.157264`
- IoU = `0.085343`

### Dibanding baseline lama

Baseline `baseline_dice_gt005_k5`:

- F1 = `0.149762`
- IoU = `0.080942`

Improved context:

- F1 = `0.157264`
- IoU = `0.085343`

### Kesimpulan

Ini adalah **peningkatan kecil yang nyata**.

Artinya:

- perubahan `wbce_dice_context` memang ada manfaatnya
- tetapi peningkatannya belum besar

Jadi fase ini termasuk:

- **keberhasilan parsial**

---

## 12. Fase Gagal: Model Menjadi “Hampir Semua Positif”

Salah satu eksperimen improved yang sangat penting justru memberi pelajaran karena gagal.

File report:

- `Ipynb/artifacts-2/convlstm_training_report_improved_context_mask_k5.json`

Metrik:

- Precision = `0.002185`
- Recall = `0.924095`
- F1-score = `0.004359`
- IoU = `0.002184`

### Apa artinya?

Sekilas recall sangat tinggi terlihat bagus.

Tetapi sebenarnya ini buruk sekali.

Kenapa?

Karena precision hampir nol.

Artinya model:

- menandai hampir semua area sebagai hotspot
- sehingga hampir semua hotspot benar ikut tertangkap
- tetapi salah alarm-nya sangat banyak

### Analogi sederhananya

Kalau ada 100 titik di peta dan model bilang:

> “Semua titik adalah hotspot”

maka:

- recall bisa tinggi
- tapi precision akan sangat jelek

Jadi model seperti ini tidak berguna secara praktik.

### Pelajaran dari fase ini

Anda belajar bahwa:

- recall tinggi saja tidak cukup
- model yang “menebak semua positif” bukan model bagus

Ini juga membuat kode training kemudian diperbaiki agar validasi tidak tertipu oleh solusi degeneratif (degeneratif = solusi salah yang terlihat bagus hanya karena satu metrik).

---

## 13. Fase Gagal: OHEM

OHEM (Online Hard Example Mining) adalah teknik agar model fokus ke contoh sulit.

Secara teori ini menarik.

Tetapi pada project Anda, eksperimen OHEM bermasalah.

### Error yang terjadi

Di `context_prompt.txt` tercatat error:

- `InvalidArgumentError`
- terkait `TopKV2_grad/range`
- berkaitan dengan XLA compile

### Arti sederhananya

Teknik OHEM yang dipakai di loss function tidak cocok dengan cara TensorFlow/Keras mencoba meng-compile graph (graph = bentuk eksekusi komputasi model).

### Apa yang dilakukan untuk memperbaiki?

Kode kemudian diperbaiki:

- `jit_compile=False`

Tujuannya:

- mencegah TensorFlow memaksa XLA pada loss custom OHEM

### Tetapi apakah hasil OHEM akhirnya bagus?

Tidak.

Dari `hasil-model-07042026.txt`, terlihat model justru jatuh ke kondisi sangat buruk:

- Precision = `0.002407`
- Recall = `0.974864`
- F1-score = `0.004803`
- IoU = `0.002407`

### Artinya

Sama seperti kasus degeneratif sebelumnya:

- model hampir menandai semua area sebagai positif

### Kesimpulan fase ini

Ini termasuk **kegagalan eksperimen**.

Tetapi tetap berguna, karena menunjukkan:

- teknik yang secara teori bagus belum tentu cocok untuk data Anda

---

## 14. Fase Error Shape pada Improved 3-Channel

Saat mencoba `feature_stack = mask_context3`, muncul error shape (shape = ukuran / dimensi tensor).

### Masalahnya

Input dibuat menjadi `3 channel`, tetapi target sempat ikut terbentuk sebagai `3 channel`, padahal output model tetap `1 channel`.

Akibatnya:

- `y_true` dan `y_pred` tidak cocok ukurannya

### Apa efeknya?

Training gagal dengan error tensor shape.

### Apa yang dilakukan?

Kode diperbaiki agar:

- input boleh `3 channel`
- target tetap `1 channel`

### Pelajaran

Ini menunjukkan bahwa:

- tidak semua kegagalan berasal dari kualitas model
- ada juga kegagalan murni karena bug teknik di pipeline

---

## 15. Fase Native Patch 1528x773

Setelah lama mencoba grid `128x128`, muncul kesadaran penting:

- titik hotspot merah terlalu kecil
- kemungkinan hilang saat resize

Karena itu dibuat file baru:

- `Ipynb/TRAIN_CONVLSTM_NATIVE_PATCH_1528x773_GOOGLE_COLAB.py`
- `Ipynb/TRAIN_CONVLSTM_NATIVE_PATCH_1528x773_GOOGLE_COLAB.ipynb`

### Tujuan native patch

Bukan memproses full-frame besar secara langsung, tetapi:

- mengambil patch (patch = potongan gambar kecil)
- dari gambar asli `1528x773`
- sehingga titik hotspot kecil tetap lebih terlihat

### Kenapa tidak full-frame langsung?

Karena:

- full-frame `1528x773` dengan ConvLSTM terlalu berat
- terutama untuk Colab atau laptop RTX 3050 6 GB

Jadi pendekatan patch adalah kompromi yang lebih realistis.

---

## 16. Fase Gagal: Broadcast Shape pada Native Patch

Saat native patch pertama kali dijalankan, muncul error:

- `required broadcastable shapes`

### Penyebab

Di loss `wbce_dice_context`:

- `context_weight` berbentuk `4D`
- `binary_crossentropy(...)` mengembalikan `3D`

Ketika dikalikan:

- shape tidak cocok

### Perbaikan

Kode kemudian diperbaiki agar:

- `context_bce_map` dihitung manual dalam bentuk `4D`

Setelah itu:

- sanity check lokal 1 batch berhasil

Ini contoh kegagalan yang akhirnya berhasil diperbaiki.

---

## 17. Fase Dukungan Run Lokal

Setelah native patch dibuat, file juga diubah agar bisa dijalankan lokal.

Perubahan penting:

- mendukung path lokal
- output diarahkan ke repo lokal
- batch default dibuat lebih aman
- dibuat launcher `.bat`

File terkait:

- `run_train_native_patch_local.bat`
- `TUTORIAL_RUN_LOCAL_TRAIN_CONVLSTM_NATIVE_PATCH_1528x773.md`

### Masalah yang sempat muncul

Anda sempat menjalankan:

- `Activate.ps1`

di `cmd`, bukan di PowerShell.

Akibatnya:

- `.venv` tidak benar-benar aktif
- `python` tidak ditemukan

### Perbaikan

Dokumentasi diperjelas:

- untuk `cmd`, gunakan `activate.bat`
- atau langsung panggil:
  - `.venv\Scripts\python.exe`

Ini kegagalan workflow, bukan kegagalan model.

---

## 18. Fase Cek GPU Lokal dan WSL2

Anda kemudian mencoba menjalankan training lokal dengan GPU.

### Hasil cek

GPU NVIDIA memang ada:

- RTX 3050 Laptop GPU

Driver juga aktif:

- `nvidia-smi` berjalan normal

Tetapi TensorFlow lokal:

- `build_cuda False`
- `gpus []`

Artinya:

- TensorFlow yang terpasang masih CPU-only

### Kenapa?

Karena environment `.venv` memakai:

- `tensorflow`
- `tensorflow-intel`

Bukan TensorFlow GPU via CUDA/WSL2.

### Status WSL2

Pemeriksaan menunjukkan:

- WSL2 belum siap penuh
- awalnya virtualisasi BIOS belum benar

### Kabar baik

Anda kemudian masuk BIOS, dan:

- `Intel Virtualization Technology = Enabled`
- `VT-d = Enabled`

Ini langkah benar dan penting.

### Masalah berikutnya

Saat mengaktifkan:

- `VirtualMachinePlatform`

proses sempat macet lama.

Ini berarti bagian WSL2 masih belum sepenuhnya mulus.

### Kesimpulan fase ini

Anda sudah berhasil maju sampai tahap:

- GPU ada
- virtualisasi BIOS aktif

Tetapi Anda belum sampai:

- TensorFlow GPU lokal siap penuh

---

## 19. Rangkuman Keberhasilan Anda

Ini daftar hal yang **benar-benar berhasil Anda capai** dalam proses training model:

### Keberhasilan 1: Model berhasil dilatih

Walaupun metriknya belum tinggi, Anda sudah berhasil:

- membuat model ConvLSTM benar-benar berjalan
- menghasilkan file `.keras`
- menghasilkan report `.json`

Ini bukan hal sepele.

### Keberhasilan 2: Dataset bertambah

Anda berhasil membawa dataset dari jumlah kecil menuju sekitar `1060` frame.

Ini penting karena:

- model punya lebih banyak contoh belajar

### Keberhasilan 3: Kualitas model meningkat dari nol

Awalnya:

- Precision = `0`
- Recall = `0`
- F1 = `0`
- IoU = `0`

Lalu naik sampai:

- Precision = `0.1512`
- Recall = `0.1639`
- F1 = `0.1573`
- IoU = `0.0853`

Memang belum tinggi, tetapi ini **kemajuan nyata**.

### Keberhasilan 4: Banyak error berhasil diperbaiki

Beberapa error yang berhasil ditutup:

- argparse di Colab
- dataset path
- shape bug 3-channel
- XLA / OHEM error
- broadcastable shapes di native patch
- workflow run lokal

### Keberhasilan 5: Integrasi ke sistem web berhasil

Ini sangat penting.

Model yang Anda latih bukan hanya menjadi file diam, tetapi sudah:

- dipasang ke backend
- dipakai menghasilkan heatmap dan overlay di web

Ini nilai besar untuk skripsi implementasi.

---

## 20. Rangkuman Kegagalan dan Hambatan

Ini daftar hal yang belum berhasil atau sempat gagal:

### Kegagalan 1: Model sangat awal gagal total

Metrik semuanya nol.

### Kegagalan 2: Dataset banyak belum otomatis membuat model bagus

Jumlah data naik, tetapi kualitas masih rendah.

### Kegagalan 3: 256x256 tidak menang

Eksperimen resolusi lebih tinggi tidak mengalahkan baseline 128.

### Kegagalan 4: Precision dan recall sulit seimbang

Ada run yang precision naik tapi recall turun.

### Kegagalan 5: Solusi degeneratif

Beberapa run menghasilkan recall tinggi tapi precision/F1/IoU hancur.

### Kegagalan 6: OHEM tidak cocok

Secara praktik, OHEM tidak berhasil meningkatkan kualitas model pada dataset ini.

### Kegagalan 7: Native patch belum sampai menghasilkan model terbaik baru

Native patch sudah dibuat, tetapi belum ada bukti kuat bahwa ia sudah melampaui model terbaik 128.

### Kegagalan 8: GPU lokal belum aktif

GPU fisik ada, tetapi training lokal GPU belum benar-benar siap karena TensorFlow GPU belum terpasang dengan benar.

---

## 21. Kenapa Kualitas Model Masih Rendah?

Ini pertanyaan paling penting.

Jawaban jujurnya:

### Penyebab 1: Input terlalu miskin

Model hanya melihat:

- peta visual hotspot

Padahal dunia nyata dipengaruhi oleh:

- hujan
- angin
- suhu
- kelembapan
- gambut
- tutupan lahan

### Penyebab 2: Label terlalu lemah

Target dibuat dari warna merah di JPG, bukan dari data hotspot mentah yang lebih bersih.

### Penyebab 3: Titik hotspot terlalu kecil

Saat gambar dikecilkan:

- titik merah bisa hilang

### Penyebab 4: Class imbalance ekstrem

Di report terbaik:

- `positive_ratio_train ≈ 0.00045495`

Artinya:

- piksel hotspot sangat sedikit dibanding background

### Penyebab 5: Masalah yang diprediksi memang sulit

Hotspot adalah fenomena nyata yang dipengaruhi banyak faktor.

Jadi memprediksinya hanya dari:

- gambar titik merah masa lalu

memang sangat membatasi model.

---

## 22. Posisi Project Saat Ini Secara Jujur

Kalau dinilai dengan jujur:

### Dari sisi web

- **cukup baik sampai baik**

### Dari sisi model

- **tidak jelek**
- **belum cukup baik**

### Dari sisi skripsi

- **sangat layak sebagai skripsi implementasi**

### Dari sisi solusi operasional nyata

- **belum cukup kuat**

---

## 23. Apa yang Sudah Anda Pelajari dari Semua Percobaan Ini

Hal-hal yang sekarang sudah terbukti dari project Anda:

1. menambah dataset membantu, tetapi tidak otomatis menyelesaikan masalah
2. resolusi lebih tinggi tidak otomatis lebih baik
3. precision, recall, F1, dan IoU harus dilihat bersama
4. model bisa terlihat “bagus” padahal sebenarnya rusak jika hanya melihat satu metrik
5. bug pipeline bisa sama merusaknya dengan kelemahan model
6. integrasi model ke web adalah keberhasilan besar tersendiri
7. bottleneck terbesar ada pada kualitas representasi data, bukan hanya bentuk model

---

## 24. Kesimpulan Akhir

Perjalanan training model Anda **bukan perjalanan gagal**.

Yang benar adalah:

- Anda sudah berhasil membangun pipeline training
- Anda sudah berhasil membuat model berjalan
- Anda sudah berhasil meningkatkan kualitas dari nol menjadi baseline yang masuk akal
- Anda sudah berhasil mengintegrasikan hasil ke web
- Anda juga sudah berhasil mengidentifikasi masalah utama model secara semakin jelas

Tetapi memang benar:

- kualitas model masih rendah
- beberapa eksperimen besar belum berhasil mengangkat model menjadi kuat

Kalau dirangkum dalam satu kalimat:

> Anda sudah berhasil membangun model ConvLSTM yang nyata dan berjalan, tetapi perjalanan eksperimen menunjukkan bahwa kualitas model masih tertahan oleh keterbatasan data, ukuran hotspot yang kecil, dan kompleksitas masalah hotspot itu sendiri.

---

## 25. Kalimat Penutup yang Paling Jujur untuk Anda

Kalau Anda bertanya:

> “Apakah saya sudah melakukan banyak hal dalam melatih model ini?”

Jawabannya:

- **ya, sudah sangat banyak**

Kalau Anda bertanya:

> “Apakah semua percobaan saya gagal?”

Jawabannya:

- **tidak**
- banyak percobaan Anda justru sangat berguna, karena:
  - ada yang berhasil meningkatkan model
  - ada yang membuktikan jalur tertentu tidak efektif
  - ada yang mengungkap bug penting

Kalau Anda bertanya:

> “Apakah saya sudah punya hasil yang cukup untuk skripsi?”

Jawabannya:

- **ya, cukup**
- asalkan dijelaskan dengan jujur sebagai:
  - baseline
  - prototype
  - hasil implementasi yang berhasil, tetapi belum sempurna

Itu posisi yang paling kuat, paling aman, dan paling masuk akal.
