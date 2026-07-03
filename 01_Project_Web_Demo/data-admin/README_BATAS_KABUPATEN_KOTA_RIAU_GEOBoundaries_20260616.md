# Batas Kabupaten/Kota Riau dari geoBoundaries ADM2

Folder ini berisi batas administrasi level kabupaten/kota untuk Provinsi Riau yang digunakan sebagai alternatif ketika file resmi BIG/Ina-Geoportal belum tersedia.

## File

| File | Keterangan |
|---|---|
| `geoBoundaries-IDN-ADM2_simplified.geojson` | Data ADM2 Indonesia dari geoBoundaries versi simplified |
| `batas_kabupaten_kota_riau_geoboundaries_adm2.geojson` | Hasil filter 12 kabupaten/kota Provinsi Riau |
| `batas_kabupaten_kota_riau_geoboundaries_adm2.metadata.json` | Metadata sumber dan keterbatasan data |

## Sumber Data

Data diambil dari geoBoundaries ADM2 Indonesia.

Endpoint metadata:

```text
https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/
```

Sumber yang tercantum pada metadata geoBoundaries:

```text
Badan Pusat Statistik (BPS - Statistics Indonesia), World Food Programme, OCHA ROAP
```

Lisensi yang tercantum pada metadata ADM2:

```text
Creative Commons Attribution 3.0 Intergovernmental Organisations (CC BY 3.0 IGO)
```

File yang digunakan:

```text
https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM2/geoBoundaries-IDN-ADM2_simplified.geojson
```

## Kabupaten/Kota yang Difilter

File `batas_kabupaten_kota_riau_geoboundaries_adm2.geojson` berisi 12 wilayah:

1. Bengkalis
2. Indragiri Hilir
3. Indragiri Hulu
4. Kampar
5. Kepulauan Meranti
6. Kota Dumai
7. Kota Pekanbaru
8. Kuantan Singingi
9. Pelalawan
10. Rokan Hilir
11. Rokan Hulu
12. Siak

## Cara Pakai untuk Spatial Join

Contoh menjalankan evaluasi geospasial dengan batas kabupaten/kota:

```powershell
python .\tools\geospatial_validate_firms_prediction.py `
  --firms-csv .\data-firms-mentah-pendamping\modis_sp_2025-06-21_2025-11-25_cakupan_peta_project.csv `
  --admin-geojson .\data-admin\batas_kabupaten_kota_riau_geoboundaries_adm2.geojson `
  --output-dir .\artifacts\geospatial_firms_validation_2025_kabupaten
```

Jika ingin mengevaluasi satu tanggal dengan probability map:

```powershell
python .\tools\geospatial_validate_firms_prediction.py `
  --firms-csv .\data-firms-mentah-pendamping\modis_sp_2025-06-21_2025-11-25_cakupan_peta_project.csv `
  --target-date 2025-07-19 `
  --probability-npy .\backend\storage\outputs\ID_PREDIKSI\probability_H+1.npy `
  --admin-geojson .\data-admin\batas_kabupaten_kota_riau_geoboundaries_adm2.geojson `
  --threshold 0.55 `
  --distance-km 1,3,5 `
  --output-dir .\artifacts\geospatial_firms_validation_2025_kabupaten
```

## Hasil Uji Awal

Data FIRMS testing 2025 pada cakupan peta project berisi:

```text
1961 titik FIRMS
```

Setelah spatial join ke batas kabupaten/kota Riau:

```text
850 titik berada pada polygon kabupaten/kota Riau
1111 titik tidak teridentifikasi sebagai Riau
```

Titik yang tidak teridentifikasi bukan berarti data salah. Penyebab utamanya adalah bounding box gambar project lebih luas daripada Provinsi Riau sehingga dapat memuat area provinsi sekitar, laut, atau titik di luar polygon Riau.

## Keterbatasan

1. File ini hanya level kabupaten/kota, bukan kecamatan.
2. File yang digunakan adalah versi simplified agar ringan.
3. Untuk analisis sangat presisi di dekat batas wilayah, gunakan data resmi atau full-resolution jika tersedia.
4. Sumber ini bukan unduhan langsung dari portal BIG/Ina-Geoportal, sehingga harus disebut sebagai alternatif sumber batas administrasi.

## Kalimat Aman untuk Laporan

> Untuk menambahkan informasi wilayah administratif, titik FIRMS dicocokkan dengan batas administrasi kabupaten/kota dari geoBoundaries ADM2 Indonesia. Data ini digunakan sebagai alternatif karena file batas administrasi resmi dalam format GeoJSON belum tersedia di project. Hasil spatial join digunakan untuk mengidentifikasi kabupaten/kota dari titik hotspot aktual, sedangkan identifikasi kecamatan belum dilakukan karena membutuhkan batas administrasi tingkat kecamatan.

