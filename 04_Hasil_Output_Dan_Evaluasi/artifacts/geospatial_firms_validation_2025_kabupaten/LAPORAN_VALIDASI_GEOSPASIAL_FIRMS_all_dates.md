# Laporan Evaluasi Geospasial FIRMS

Dokumen ini dihasilkan otomatis sebagai bukti pendamping evaluasi model.
Evaluasi ini tidak mengganti evaluasi raster/piksel yang sudah ada, tetapi
menambahkan pembuktian berbasis titik koordinat FIRMS.

## Konfigurasi

- CSV FIRMS: `data-firms-mentah-pendamping\modis_sp_2025-06-21_2025-11-25_cakupan_peta_project.csv`
- Tanggal target: `semua tanggal pada CSV`
- Probability NPY: `tidak digunakan`
- Threshold: `0.55`
- Center peta: `102.1, 0.8`
- Zoom: `8.1`
- Ukuran gambar: `1528 x 773`
- Km per piksel estimasi: `0.570491`
- Batas administrasi: `data-admin\batas_kabupaten_kota_riau_geoboundaries_adm2.geojson`

## Ringkasan Hasil

- Jumlah titik FIRMS dibaca: `1961`
- Titik di dalam cakupan gambar: `1961`
- Titik di luar cakupan gambar: `0`
- Evaluasi jarak prediksi belum dihitung karena `--probability-npy` tidak diberikan.

## File Output

- Detail titik: `artifacts\geospatial_firms_validation_2025_kabupaten\validasi_geospasial_detail_all_dates.csv`
- Ringkasan JSON: `artifacts\geospatial_firms_validation_2025_kabupaten\validasi_geospasial_summary_all_dates.json`
- Overlay: `artifacts\geospatial_firms_validation_2025_kabupaten\overlay_prediksi_vs_firms_all_dates.png`
- Rekap kabupaten: `artifacts\geospatial_firms_validation_2025_kabupaten\rekap_per_kabupaten_all_dates.csv`

## Batas Interpretasi

- Titik FIRMS merupakan pusat piksel deteksi satelit, bukan posisi api lapangan yang presisi.
- Validasi radius 1 km, 3 km, dan 5 km adalah validasi toleransi spasial, bukan validasi ground check lapangan.
- Nama kabupaten/kecamatan hanya kuat jika diperoleh dari file batas administrasi geospasial.
- Jika batas administrasi tidak diberikan, tabel tetap memuat koordinat dan atribut FIRMS, tetapi belum memuat nama wilayah administratif.
