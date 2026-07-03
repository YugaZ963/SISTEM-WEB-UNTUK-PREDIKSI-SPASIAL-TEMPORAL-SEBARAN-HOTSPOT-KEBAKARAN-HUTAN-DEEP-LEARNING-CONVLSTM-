# Laporan Evaluasi Geospasial FIRMS

Dokumen ini dihasilkan otomatis sebagai bukti pendamping evaluasi model.
Evaluasi ini tidak mengganti evaluasi raster/piksel yang sudah ada, tetapi
menambahkan pembuktian berbasis titik koordinat FIRMS.

## Konfigurasi

- CSV FIRMS: `data-firms-mentah-pendamping\modis_sp_2025-06-21_2025-11-25_cakupan_peta_project.csv`
- Tanggal target: `2025-07-19`
- Probability NPY: `backend\storage\outputs\a8a415ef-0459-4569-8669-8419c24d4c54\probability_H+1.npy`
- Threshold: `0.55`
- Center peta: `102.1, 0.8`
- Zoom: `8.1`
- Ukuran gambar: `1528 x 773`
- Km per piksel estimasi: `0.570491`
- Batas administrasi: `data-admin\batas_kabupaten_kota_riau_geoboundaries_adm2.geojson`

## Ringkasan Hasil

- Jumlah titik FIRMS dibaca: `251`
- Titik di dalam cakupan gambar: `251`
- Titik di luar cakupan gambar: `0`
- Piksel area prediksi di atas threshold: `5138`
- Persentase area prediksi: `0.00435002`
- Radius 1 km: `0` titik sesuai (`hit_rate=0.0`)
- Radius 3 km: `0` titik sesuai (`hit_rate=0.0`)
- Radius 5 km: `1` titik sesuai (`hit_rate=0.003984`)

## File Output

- Detail titik: `artifacts\validasi_geospasial_2025-07-19\validasi_geospasial_detail_2025-07-19.csv`
- Ringkasan JSON: `artifacts\validasi_geospasial_2025-07-19\validasi_geospasial_summary_2025-07-19.json`
- Overlay: `artifacts\validasi_geospasial_2025-07-19\overlay_prediksi_vs_firms_2025-07-19.png`
- Rekap kabupaten: `artifacts\validasi_geospasial_2025-07-19\rekap_per_kabupaten_2025-07-19.csv`

## Batas Interpretasi

- Titik FIRMS merupakan pusat piksel deteksi satelit, bukan posisi api lapangan yang presisi.
- Validasi radius 1 km, 3 km, dan 5 km adalah validasi toleransi spasial, bukan validasi ground check lapangan.
- Nama kabupaten/kecamatan hanya kuat jika diperoleh dari file batas administrasi geospasial.
- Jika batas administrasi tidak diberikan, tabel tetap memuat koordinat dan atribut FIRMS, tetapi belum memuat nama wilayah administratif.
