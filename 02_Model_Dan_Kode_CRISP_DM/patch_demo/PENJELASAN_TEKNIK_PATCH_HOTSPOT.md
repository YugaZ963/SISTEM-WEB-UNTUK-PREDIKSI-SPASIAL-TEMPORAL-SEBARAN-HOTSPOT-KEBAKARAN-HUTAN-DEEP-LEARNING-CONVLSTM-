# Penjelasan Teknik Patch pada Project Hotspot

Dokumen ini menjelaskan teknik **patch** (memotong citra besar menjadi potongan kecil) berdasarkan contoh nyata dari dataset project.

## 1. File Dataset yang Dipakai Sebagai Contoh

Contoh diambil dari file:

```text
Ipynb/Dataset History Fire Hotspot In Riau Province PNG/FIRMS_2025-07-19[@102.1,0.8,8.1z].png
```

Ukuran citra asli:

```text
1528 x 773 piksel
```

Konfigurasi patch:

```text
patch_width  = 160
patch_height = 160
patch_stride = 80
```

Keterangan:

- `patch_width` adalah lebar potongan citra.
- `patch_height` adalah tinggi potongan citra.
- `patch_stride` adalah jarak pergeseran patch saat sistem melakukan prediksi patch-stitch. Jika stride lebih kecil dari ukuran patch, maka antar-patch saling tumpang tindih.

## 2. Gambar Visual Teknik Patch

Lihat gambar berikut:

```text
docs/patch_demo/ringkasan_visual_teknik_patch.png
```

Isi gambar tersebut:

- bagian atas: citra penuh dengan grid patch 160x160
- kotak merah: contoh patch positif
- kotak biru: contoh patch negatif
- bagian bawah: hasil crop patch dan mask hotspot

File gambar penuh dengan grid patch:

```text
docs/patch_demo/gambar_penuh_dengan_grid_patch_160.png
```

## 3. Apa Itu Patch?

Patch adalah potongan kecil dari citra besar.

Pada project ini, citra asli berukuran besar:

```text
1528 x 773 piksel
```

Jika langsung dimasukkan ke model, proses training menjadi berat. Karena itu citra dipotong menjadi bagian kecil:

```text
160 x 160 piksel
```

Secara sederhana:

```text
citra besar
-> dipotong menjadi banyak patch kecil
-> patch yang mengandung hotspot dipakai sebagai data penting untuk training
```

## 4. Kenapa Tidak Langsung Resize Full Image ke 160x160?

Ini bagian penting.

Kalau citra besar langsung dikecilkan:

```text
1528 x 773 -> 160 x 160
```

maka titik hotspot merah yang kecil bisa hilang.

Masalahnya, hotspot di dataset ini sangat kecil dibanding area background. Jika langsung resize full image, piksel merah bisa tercampur dengan warna lain dan tidak terbaca lagi oleh model.

Karena itu cara yang lebih aman adalah:

```text
citra asli
-> ekstraksi hotspot merah
-> bentuk mask hotspot
-> crop patch 160x160
-> masuk ke model
```

## 5. Patch Positif dan Patch Negatif

Dalam contoh ini dibuat dua jenis patch:

### Patch Positif

Patch positif adalah patch yang mengandung hotspot.

Contoh file:

```text
docs/patch_demo/patch_positif_160x160_rgb.png
docs/patch_demo/patch_positif_160x160_mask.png
```

Patch positif penting karena model perlu melihat contoh area yang benar-benar memiliki hotspot.

### Patch Negatif

Patch negatif adalah patch yang tidak mengandung hotspot.

Contoh file:

```text
docs/patch_demo/patch_negatif_160x160_rgb.png
docs/patch_demo/patch_negatif_160x160_mask.png
```

Patch negatif tetap dibutuhkan supaya model juga belajar membedakan area yang tidak berisiko.

Namun jumlah patch negatif tidak boleh terlalu dominan, karena dataset hotspot sudah sangat tidak seimbang.

## 6. Kenapa Patch Positif Lebih Penting?

Pada dataset hotspot, area merah sangat sedikit.

Dari contoh file ini:

```text
raw_red_pixel_count     = 4288
dilated_red_pixel_count = 9099
```

Artinya, walaupun gambar penuh sangat besar, hanya sebagian kecil piksel yang benar-benar termasuk hotspot.

Karena itu saat training, sistem tidak cukup mengambil patch secara acak. Kalau patch diambil acak, kemungkinan besar yang terambil adalah patch kosong tanpa hotspot.

Strategi yang lebih baik:

```text
ambil beberapa patch positif
ambil sedikit patch negatif
```

Contoh konfigurasi training yang pernah dipakai:

```text
train-positive-patches = 4
train-negative-patches = 1
```

Artinya, untuk setiap sample, sistem lebih banyak mengambil patch yang mengandung hotspot dibanding patch kosong.

## 7. Hubungan Patch dengan ConvLSTM

ConvLSTM membaca data dalam bentuk sequence.

Pada project ini, input model adalah:

```text
H-6, H-5, H-4, H-3, H-2, H-1, H0
```

Targetnya:

```text
H+1
```

Jika memakai patch, maka konsepnya menjadi:

```text
patch area yang sama dari H-6
patch area yang sama dari H-5
patch area yang sama dari H-4
patch area yang sama dari H-3
patch area yang sama dari H-2
patch area yang sama dari H-1
patch area yang sama dari H0
-> prediksi risk map patch H+1
```

Jadi model tidak membaca satu gambar saja. Model membaca perubahan patch dari beberapa hari.

## 8. Perbedaan Patch Training dan Patch-Stitch di Web

Pada training:

```text
citra besar -> dipilih patch positif dan negatif -> model belajar
```

Pada sistem web:

```text
citra besar -> dipotong menjadi banyak patch -> setiap patch diprediksi -> hasil patch digabung lagi
```

Proses menggabungkan kembali patch disebut **patch-stitch** (menjahit kembali hasil prediksi patch menjadi satu peta besar).

Project memakai:

```text
patch_size   = 160
patch_stride = 80
```

Karena stride 80 lebih kecil dari patch 160, patch saling tumpang tindih. Ini membantu mengurangi garis patahan antar-patch saat hasil prediksi digabung.

## 9. Kesimpulan

Teknik patch pada project ini digunakan karena:

- citra asli besar
- hotspot kecil dan langka
- model lebih mudah belajar dari area lokal
- training lebih ringan
- patch positif bisa dipilih agar model lebih sering melihat hotspot

Namun patch tidak boleh dipahami sebagai sekadar mengecilkan gambar.

Yang benar:

```text
patch = memotong bagian citra asli
```

Bukan:

```text
patch = mengecilkan seluruh citra menjadi kecil
```

Kalau seluruh citra dikecilkan langsung ke 160x160, titik hotspot bisa hilang. Karena itu pada project ini, pendekatan patch lebih tepat daripada resize full image.

