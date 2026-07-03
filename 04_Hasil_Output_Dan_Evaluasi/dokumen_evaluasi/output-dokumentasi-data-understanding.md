Saya sudah buat script terpisah khusus untuk **Data Understanding**:

- [data_understanding_historical_risk_patch.py](C:/laragon/www/sistem-web-skripsi-ta/tools/data_understanding_historical_risk_patch.py)

**Isi script ini**

- pembacaan dataset dan parsing tanggal: [data_understanding_historical_risk_patch.py:38](C:/laragon/www/sistem-web-skripsi-ta/tools/data_understanding_historical_risk_patch.py:38)
- validasi file gambar: [data_understanding_historical_risk_patch.py:51](C:/laragon/www/sistem-web-skripsi-ta/tools/data_understanding_historical_risk_patch.py:51)
- pembentukan sequence: [data_understanding_historical_risk_patch.py:66](C:/laragon/www/sistem-web-skripsi-ta/tools/data_understanding_historical_risk_patch.py:66)
- split kronologis: [data_understanding_historical_risk_patch.py:70](C:/laragon/www/sistem-web-skripsi-ta/tools/data_understanding_historical_risk_patch.py:70)
- perhitungan class imbalance opsional: [data_understanding_historical_risk_patch.py:124](C:/laragon/www/sistem-web-skripsi-ta/tools/data_understanding_historical_risk_patch.py:124)
- ringkasan output terminal: [data_understanding_historical_risk_patch.py:194](C:/laragon/www/sistem-web-skripsi-ta/tools/data_understanding_historical_risk_patch.py:194) dan [data_understanding_historical_risk_patch.py:249](C:/laragon/www/sistem-web-skripsi-ta/tools/data_understanding_historical_risk_patch.py:249)

**Kenapa saya buat begini**

- script ini **tidak menjalankan training**
- script ini **tidak import TensorFlow**
- script ini hanya fokus ke:
  - baca dataset
  - validasi file
  - split kronologis
  - ringkasan dataset
  - opsional hitung `positive mass ratio`

Itu lebih cocok untuk kebutuhan screenshot laporan.

**Command yang dipakai**

Mode cepat untuk screenshot Data Understanding:

```powershell
python tools/data_understanding_historical_risk_patch.py --dataset-dir "Ipynb/Dataset History Fire Hotspot In Riau Province PNG" --image-extensions .png
```

Mode lengkap kalau Anda juga ingin angka `Positive mass ratio` dan `Positive class weight`:

```powershell
python tools/data_understanding_historical_risk_patch.py --dataset-dir "Ipynb/Dataset History Fire Hotspot In Riau Province PNG" --image-extensions .png --with-positive-weight
```

Mode JSON kalau Anda ingin salin hasil lebih rapi:

```powershell
python tools/data_understanding_historical_risk_patch.py --dataset-dir "Ipynb/Dataset History Fire Hotspot In Riau Province PNG" --image-extensions .png --json
```

**Output cepat yang sudah terverifikasi**
Mode default menghasilkan ringkasan seperti:

- `Dataset ditemukan: 1060 frame valid`
- `Rentang data: 2023-01-01 s.d. 2025-11-25`
- `Total sample sequence: 1053`
- `Train samples: 737`
- `Val samples: 158`
- `Test samples: 158`

**Catatan penting**

- Saya sengaja membuat `positive mass ratio` menjadi **opsional** lewat `--with-positive-weight`.
- Alasannya sederhana: menghitung nilai itu tetap harus membaca seluruh target train, jadi lebih berat daripada ringkasan dataset biasa.
- Untuk screenshot laporan, pendekatan yang lebih cerdas adalah:
  1. pakai mode cepat untuk SS struktur data
  2. pakai mode `--with-positive-weight` hanya jika Anda benar-benar perlu SS angka imbalance exact

**Verifikasi**

- `python -m py_compile tools/data_understanding_historical_risk_patch.py` lolos
- `--help` lolos
- mode cepat berhasil jalan

Kalau mau, saya bisa lanjut buat:

1. script terpisah yang sama untuk **Data Preparation**
2. script terpisah untuk **Evaluation**
3. panduan **persis bagian mana yang di-crop** dari file baru ini untuk screenshot laporan
