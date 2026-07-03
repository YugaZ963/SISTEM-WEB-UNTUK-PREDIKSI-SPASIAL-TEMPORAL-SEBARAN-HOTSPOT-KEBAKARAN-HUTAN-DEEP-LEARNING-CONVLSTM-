# Data FIRMS Mentah Pendamping

Folder ini berisi data atribut dan vektor FIRMS yang digunakan sebagai
pendamping evaluasi geospasial. Data ini tidak mengganti dataset PNG, model,
training, atau backend sistem web.

## Sumber Resmi

- NASA FIRMS Archive Download:
  https://firms.modaps.eosdis.nasa.gov/download/
- NASA FIRMS Area API:
  https://firms.modaps.eosdis.nasa.gov/api/area/
- NASA FIRMS FAQ:
  https://www.earthdata.nasa.gov/data/tools/firms/faq

Produk yang dipilih adalah MODIS Collection 6.1 Standard Processing
(`MODIS_SP`) karena sesuai dengan satelit Terra/Aqua dan lebih tepat untuk
evaluasi historis dibandingkan produk NRT.

## File Pilot yang Sudah Diunduh

- `modis_2024_Indonesia.csv`: data MODIS resmi Indonesia tahun 2024.
- `modis_2024_cakupan_peta_project.csv`: data yang difilter ke cakupan peta
  project.
- `modis_2024-10-27_cakupan_peta_project.csv`: 81 titik untuk tanggal pilot.
- `overlay_modis_2024-10-27_vs_png.png`: overlay koordinat MODIS di atas PNG.
- `kecocokan_modis_2024-10-27_vs_png.csv`: posisi piksel dan jarak ke piksel
  merah terdekat.

## Hasil Pilot

Uji dilakukan terhadap:

```text
PNG: FIRMS_2024-10-27[@102.1,0.8,8.1z].png
Pusat peta: 102.1 BT, 0.8 LU
Zoom: 8.1
Ukuran: 1528 x 773 piksel
```

Hasil:

```text
Titik MODIS dalam cakupan gambar: 81
Titik dalam radius 3 piksel dari penanda merah: 81
Median jarak ke piksel merah terdekat: 0,39 piksel
```

Hasil ini membuktikan bahwa penanda merah pada PNG tanggal pilot selaras
dengan koordinat titik MODIS resmi. Pilot ini belum menggantikan evaluasi
independen pada data test tahun 2025.

## Mengunduh Periode Testing 2025

1. Minta MAP_KEY gratis:
   https://firms.modaps.eosdis.nasa.gov/api/map_key/

2. Buka PowerShell dari root project.

3. Simpan key hanya untuk sesi terminal aktif:

```powershell
$env:FIRMS_MAP_KEY="MAP_KEY_ANDA"
```

4. Jalankan:

```powershell
python .\tools\download_firms_modis_evaluation.py
```

Konfigurasi default:

```text
Produk: MODIS_SP
Tanggal: 2025-06-21 sampai 2025-11-25
Bounding box: 98.1843,-1.1808,106.0157,2.7799
```

Skrip menghasilkan satu CSV gabungan dan satu file metadata JSON. MAP_KEY
tidak ditulis ke CSV, metadata, atau source code.

## Batas Interpretasi

Koordinat MODIS merepresentasikan pusat piksel aktif api dengan resolusi
sekitar 1 km, bukan lokasi api yang presisi. Karena itu, validasi sebaiknya
menggunakan toleransi jarak atau area risiko, bukan menuntut kecocokan titik
tanpa toleransi.
