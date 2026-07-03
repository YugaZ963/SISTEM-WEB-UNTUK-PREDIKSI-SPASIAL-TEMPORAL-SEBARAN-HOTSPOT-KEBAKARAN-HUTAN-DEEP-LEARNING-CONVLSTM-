# Kajian Batas Administrasi dan Spatial Join FIRMS

Dokumen ini menjawab catatan terkait:

1. file batas administrasi kecamatan/kabupaten resmi;
2. nama kecamatan dari koordinat hotspot;
3. opsi implementasi yang tidak membuat project menjadi terlalu rumit.

Catatan keamanan: `FIRMS_MAP_KEY` hanya digunakan sebagai variabel environment saat mengunduh data dari API. Key tidak boleh ditulis ke source code, file laporan, file `.md`, CSV, atau metadata.

## Ringkasan Keputusan

Solusi terbaik untuk project ini adalah membuat evaluasi geospasial sebagai modul pendamping, bukan mengubah model ConvLSTM.

Alur yang disarankan:

```text
Data FIRMS MODIS_SP testing
        ↓
Titik latitude-longitude + atribut sensor
        ↓
Spatial join dengan batas administrasi
        ↓
Nama provinsi, kabupaten, kecamatan
        ↓
Overlay dengan hasil prediksi
        ↓
Evaluasi jarak 1 km, 3 km, dan 5 km
```

Tahapan ini cukup untuk memperkuat skripsi karena menjawab dua hal yang belum ada pada pipeline PNG: data atribut dan data vektor.

## Kondisi Project Saat Ini

Berdasarkan pengecekan project, dataset utama masih berupa gambar PNG/JPG hasil tangkapan peta FIRMS. File tersebut memiliki tanggal dan tampilan peta, tetapi tidak membawa atribut titik hotspot seperti `latitude`, `longitude`, `confidence`, `frp`, dan nama wilayah.

Contoh data yang ada:

```text
FIRMS_2024-10-27[@102.1,0.8,8.1z].png
```

Artinya:

- tanggal tersedia dari nama file;
- pusat peta dan zoom dapat diperkirakan dari nama file;
- titik merah dapat diekstraksi dari piksel;
- tetapi koordinat titik hotspot dan nama kecamatan tidak tersimpan dalam PNG.

## Data FIRMS sebagai Data Atribut dan Vektor

NASA FIRMS Area API menyediakan data aktif api dalam format CSV dengan parameter `MAP_KEY`, `SOURCE`, `AREA_COORDINATES`, `DAY_RANGE`, dan tanggal. Dokumentasi FIRMS menyebut endpoint:

```text
/api/area/csv/[MAP_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]/[DATE]
```

Produk yang sesuai dengan batasan penelitian adalah:

```text
MODIS_SP
```

`MODIS_SP` adalah MODIS Standard Processing. FIRMS Area API juga menjelaskan bahwa `DAY_RANGE` berada pada rentang 1 sampai 5 hari per request.

Sumber:

- FIRMS Area API: https://firms.modaps.eosdis.nasa.gov/api/area/
- NASA FIRMS: https://www.earthdata.nasa.gov/data/tools/firms

Kolom penting dari CSV FIRMS:

| Kolom | Fungsi |
|---|---|
| `latitude` | Koordinat lintang titik hotspot |
| `longitude` | Koordinat bujur titik hotspot |
| `acq_date` | Tanggal pengamatan |
| `acq_time` | Waktu pengamatan |
| `satellite` | Satelit, misalnya Terra atau Aqua |
| `instrument` | Instrumen, yaitu MODIS |
| `confidence` | Tingkat keyakinan deteksi |
| `brightness` | Suhu kecerahan |
| `frp` | Fire Radiative Power |
| `daynight` | Pengamatan siang atau malam |

Data ini sudah dapat disebut sebagai:

- data atribut, karena setiap titik memiliki informasi sensor dan pengamatan;
- data vektor, karena setiap titik memiliki koordinat `longitude` dan `latitude`.

## Masalah Nama Kabupaten dan Kecamatan

CSV FIRMS tidak langsung memuat nama kabupaten atau kecamatan. Nama wilayah tidak boleh ditebak dari koordinat secara manual.

Nama wilayah harus ditambahkan melalui:

```text
Titik FIRMS latitude-longitude
        +
Poligon batas administrasi
        ↓
Spatial join / point-in-polygon
        ↓
Nama kabupaten dan kecamatan
```

Tanpa file batas administrasi, hasil seperti "titik ini berada di Kecamatan X" belum kuat secara akademik.

## Sumber Batas Administrasi

### 1. BIG / Ina-Geoportal

Sumber prioritas adalah Badan Informasi Geospasial melalui Ina-Geoportal/Tanah Air Indonesia.

Link:

```text
https://tanahair.indonesia.go.id/portal-web/
```

Kelebihan:

- sumber pemerintah;
- paling kuat untuk laporan akademik;
- sesuai dengan konsep satu peta.

Kekurangan:

- portal berbasis web interaktif;
- proses unduh bisa lebih sulit;
- perlu memastikan layer batas administrasi tersedia sampai tingkat kabupaten atau kecamatan.

Jika data dari BIG berhasil diperoleh, gunakan sebagai sumber utama.

### 2. Geoportal Pemerintah Provinsi Riau

Jika tersedia, Geoportal Riau dapat digunakan sebagai sumber batas wilayah lokal.

Kelebihan:

- fokus wilayah Riau;
- lebih relevan untuk project ini.

Kekurangan:

- ketersediaan unduhan perlu dicek;
- format dan lisensi harus diverifikasi;
- bisa saja hanya menyediakan tampilan peta, bukan unduhan GeoJSON/Shapefile.

### 3. GADM

GADM dapat digunakan sebagai alternatif akademik apabila data resmi pemerintah sulit diunduh.

Sumber:

- Download GADM: https://gadm.org/download_country.html
- Informasi data GADM: https://gadm.org/data.html
- Lisensi GADM: https://gadm.org/license.html

Catatan dari sumber GADM:

- versi saat ini adalah 4.1;
- data tersedia per negara;
- data tersedia untuk penggunaan akademik dan non-komersial;
- redistribusi atau penggunaan komersial tidak diperbolehkan tanpa izin.

Kelebihan:

- lebih mudah digunakan;
- tersedia sebagai data geospasial;
- cukup untuk spatial join dalam skripsi akademik.

Kekurangan:

- bukan sumber resmi pemerintah Indonesia;
- perlu mencantumkan sumber dan batas lisensi;
- nama dan batas wilayah perlu dicek ulang terhadap konteks Indonesia.

Rekomendasi: gunakan GADM hanya sebagai alternatif jika data BIG/Ina-Geoportal atau Geoportal Riau tidak bisa diperoleh dalam waktu yang realistis.

### 4. BPS

BPS kuat untuk tabel kode wilayah, nama kecamatan, dan statistik wilayah, tetapi umumnya bukan sumber utama geometri batas poligon.

Kegunaan BPS:

- verifikasi nama kabupaten/kecamatan;
- verifikasi kode wilayah;
- mendukung tabel administrasi.

Keterbatasan:

- tabel BPS tidak otomatis menyediakan poligon batas;
- tidak cukup untuk spatial join tanpa geometri.

## Tentang Reverse Geocoding

Reverse geocoding, misalnya memakai OpenStreetMap/Nominatim, dapat memberi nama wilayah dari koordinat. Namun cara ini sebaiknya hanya digunakan untuk pengecekan cepat atau contoh awal.

Kelemahan reverse geocoding:

- hasil tidak selalu lengkap sampai kecamatan;
- bergantung pada data pihak ketiga;
- hasil dapat berubah sewaktu-waktu;
- sulit direproduksi untuk ribuan titik;
- kurang kuat dibanding spatial join dengan batas administrasi.

Kesimpulan: reverse geocoding tidak disarankan sebagai metode final untuk skripsi.

## Ide Overlay Garis Batas di Peta

Ide menampilkan garis batas kecamatan/kabupaten di foreground peta sangat baik untuk visualisasi. Namun ada perbedaan penting:

| Bentuk Garis Batas | Bisa untuk Visual | Bisa untuk Bukti Wilayah |
|---|---:|---:|
| Garis hanya sebagai gambar | Ya | Tidak kuat |
| Garis dari GeoJSON/Shapefile | Ya | Ya |

Jadi, sebaiknya gunakan file batas administrasi geospasial asli. File yang sama dapat dipakai untuk:

1. menggambar garis batas di overlay;
2. melakukan spatial join;
3. membuat rekap per kabupaten/kecamatan.

## Implementasi Minimum yang Disarankan

Implementasi paling aman dan tidak terlalu rumit:

1. Unduh FIRMS `MODIS_SP` untuk periode testing 2025.
2. Gunakan titik FIRMS sebagai data vektor aktual.
3. Gunakan atribut FIRMS sebagai tabel atribut hotspot.
4. Tambahkan batas administrasi kabupaten terlebih dahulu.
5. Lakukan spatial join ke kabupaten.
6. Tambahkan kecamatan jika data batas kecamatan tersedia.
7. Buat overlay prediksi, titik aktual, dan batas wilayah.
8. Hitung jarak titik aktual ke area prediksi pada radius 1 km, 3 km, dan 5 km.

Prioritas implementasi:

```text
Prioritas 1: validasi titik FIRMS vs area prediksi
Prioritas 2: rekap per kabupaten
Prioritas 3: rekap per kecamatan
```

Jika kecamatan sulit, jangan memaksakan. Rekap kabupaten sudah cukup untuk menjawab kritik awal dengan lebih aman.

## Output yang Perlu Dibuat

Output pendamping yang disarankan:

```text
data-firms-mentah-pendamping/
├── modis_sp_testing_2025.csv
├── firms_testing_2025_riau_enriched.csv
├── geospatial_match_detail.csv
├── geospatial_summary.json
├── summary_per_kabupaten.csv
├── summary_per_kecamatan.csv
├── overlay_prediction_actual/
│   ├── overlay_2025-07-19.png
│   ├── overlay_2025-08-22.png
│   └── overlay_2025-10-30.png
└── LAPORAN_VALIDASI_GEOSPASIAL_FIRMS.md
```

Kolom ideal untuk hasil akhir:

| Kolom | Isi |
|---|---|
| `acq_date` | Tanggal hotspot aktual |
| `latitude` | Lintang titik aktual |
| `longitude` | Bujur titik aktual |
| `satellite` | Terra/Aqua |
| `confidence` | Nilai confidence |
| `frp` | Fire Radiative Power |
| `provinsi` | Nama provinsi hasil spatial join |
| `kabupaten` | Nama kabupaten hasil spatial join |
| `kecamatan` | Nama kecamatan hasil spatial join, jika tersedia |
| `probabilitas_prediksi` | Nilai probabilitas model di titik aktual |
| `jarak_prediksi_km` | Jarak ke area prediksi terdekat |
| `status_1km` | Sesuai/tidak pada radius 1 km |
| `status_3km` | Sesuai/tidak pada radius 3 km |
| `status_5km` | Sesuai/tidak pada radius 5 km |

## Batas Klaim dalam Laporan

Klaim yang aman:

> Data FIRMS mentah digunakan sebagai data atribut dan data vektor pendamping untuk memperkuat evaluasi hasil prediksi. Titik aktual FIRMS dibandingkan dengan area risiko hasil model berdasarkan posisi geografis dan toleransi jarak.

Klaim yang perlu dihindari:

> Model divalidasi langsung terhadap kejadian kebakaran lapangan.

Alasannya, FIRMS adalah deteksi satelit, bukan ground check lapangan. Selain itu, titik FIRMS merepresentasikan pusat piksel aktif api, bukan titik api presisi tanpa kesalahan lokasi.

## Kesimpulan

Solusi terbaik adalah menambahkan data FIRMS mentah dan batas administrasi sebagai modul evaluasi geospasial pendamping.

Jika data batas resmi dari BIG/Ina-Geoportal tersedia, gunakan itu sebagai sumber utama. Jika tidak, gunakan GADM sebagai alternatif akademik dengan menyebutkan sumber dan lisensinya. Untuk menjaga scope skripsi tetap terkendali, mulai dari rekap kabupaten terlebih dahulu, kemudian lanjut ke kecamatan hanya jika data geometri kecamatan tersedia.

Pendekatan ini tidak mengubah model, tidak mengulang training, dan tidak merusak sistem web yang sudah ada. Fungsinya adalah memperkuat pembuktian bahwa hasil prediksi dapat dibandingkan dengan titik hotspot aktual berbasis koordinat dan wilayah administrasi.
