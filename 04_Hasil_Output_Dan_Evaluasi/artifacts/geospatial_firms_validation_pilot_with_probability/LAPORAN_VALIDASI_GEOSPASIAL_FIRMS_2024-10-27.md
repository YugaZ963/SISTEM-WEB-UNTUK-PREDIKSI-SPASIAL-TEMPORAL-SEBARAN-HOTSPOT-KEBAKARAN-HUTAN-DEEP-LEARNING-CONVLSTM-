# Laporan Evaluasi Geospasial FIRMS

Dokumen ini dihasilkan otomatis sebagai bukti pendamping evaluasi model.
Evaluasi ini tidak mengganti evaluasi raster/piksel yang sudah ada, tetapi
menambahkan pembuktian berbasis titik koordinat FIRMS.

## Konfigurasi

- CSV FIRMS: `data-firms-mentah-pendamping\modis_2024-10-27_cakupan_peta_project.csv`
- Tanggal target: `2024-10-27`
- Probability NPY: `artifacts\geospatial_firms_validation_pilot\dummy_probability_from_png_redmask_2024-10-27.npy`
- Threshold: `0.55`
- Center peta: `102.1, 0.8`
- Zoom: `8.1`
- Ukuran gambar: `1528 x 773`
- Km per piksel estimasi: `0.570491`
- Batas administrasi: `tidak digunakan`

## Ringkasan Hasil

- Jumlah titik FIRMS dibaca: `81`
- Titik di dalam cakupan gambar: `81`
- Titik di luar cakupan gambar: `0`
- Piksel area prediksi di atas threshold: `7376`
- Persentase area prediksi: `0.00624479`
- Radius 1 km: `81` titik sesuai (`hit_rate=1.0`)
- Radius 3 km: `81` titik sesuai (`hit_rate=1.0`)
- Radius 5 km: `81` titik sesuai (`hit_rate=1.0`)

## File Output

- Detail titik: `artifacts\geospatial_firms_validation_pilot_with_probability\validasi_geospasial_detail_2024-10-27.csv`
- Ringkasan JSON: `artifacts\geospatial_firms_validation_pilot_with_probability\validasi_geospasial_summary_2024-10-27.json`
- Overlay: `artifacts\geospatial_firms_validation_pilot_with_probability\overlay_prediksi_vs_firms_2024-10-27.png`

## Batas Interpretasi

- Titik FIRMS merupakan pusat piksel deteksi satelit, bukan posisi api lapangan yang presisi.
- Validasi radius 1 km, 3 km, dan 5 km adalah validasi toleransi spasial, bukan validasi ground check lapangan.
- Nama kabupaten/kecamatan hanya kuat jika diperoleh dari file batas administrasi geospasial.
- Jika batas administrasi tidak diberikan, tabel tetap memuat koordinat dan atribut FIRMS, tetapi belum memuat nama wilayah administratif.
