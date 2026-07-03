"""Komentar file skripsi:
Manajemen storage prediksi untuk menyimpan input, output, metadata, dan riwayat prediksi.

Konteks laporan: file ini mendukung BAB IV pada bagian implementasi/deployment sistem,
karena setiap proses prediksi harus memiliki folder input, folder output, metadata params.json,
serta URL file hasil agar dapat ditampilkan kembali pada dashboard dan riwayat prediksi.
"""

# json digunakan untuk menyimpan dan membaca metadata prediksi dalam format params.json.
import json

# shutil digunakan untuk menyalin isi file upload dari stream FastAPI ke folder storage.
import shutil

# Path digunakan agar pengelolaan lokasi file/folder lebih rapi dan aman lintas sistem operasi.
from pathlib import Path

# settings berisi konfigurasi folder storage, input, output, dan path publik sistem.
from backend import settings

# ValidatedUpload adalah struktur data hasil validasi file dari validator.py.
from backend.services.validator import ValidatedUpload


# Fungsi untuk membuat folder input dan output berdasarkan prediction_id.
def create_prediction_dirs(prediction_id: str) -> tuple[Path, Path]:
    # Menentukan folder input khusus untuk satu proses prediksi.
    input_dir = settings.INPUTS_DIR / prediction_id

    # Menentukan folder output khusus untuk satu proses prediksi.
    output_dir = settings.OUTPUTS_DIR / prediction_id

    # Membuat folder input secara rekursif; tidak error jika folder sudah ada.
    input_dir.mkdir(parents=True, exist_ok=True)

    # Membuat folder output secara rekursif; tidak error jika folder sudah ada.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mengembalikan path folder input dan output agar dipakai oleh proses prediksi berikutnya.
    return input_dir, output_dir


# Fungsi asynchronous untuk menyimpan 7 file upload yang sudah divalidasi dan diurutkan.
async def save_upload_files(
    # ordered_files berisi file H-6 sampai H0 yang sudah lolos validasi dari validator.py.
    ordered_files: list[ValidatedUpload],
    # input_dir adalah folder tujuan penyimpanan file input untuk prediction_id tertentu.
    input_dir: Path,
) -> dict[str, Path]:
    # Dictionary untuk menyimpan relasi stem H-6/H-5/.../H0 dengan path file yang tersimpan.
    saved_paths: dict[str, Path] = {}

    # Melakukan iterasi setiap file yang sudah divalidasi dan diurutkan.
    for record in ordered_files:
        # Mengambil object upload asli dari hasil validasi.
        upload = record.upload

        # Mengambil ekstensi file asli; jika kosong, sistem memakai .png sebagai fallback.
        suffix = Path(upload.filename or "").suffix.lower() or ".png"

        # Jika tanggal akuisisi tersedia, nama file disimpan dengan format H-6_YYYY-MM-DD.ext.
        if record.acquired_date is not None:
            # Membentuk nama file bertanggal agar riwayat input mudah dilacak.
            filename = f"{record.stem}_{record.acquired_date.isoformat()}{suffix}"

        # Jika tanggal tidak tersedia, nama file disimpan dengan format H-6.ext.
        else:
            # Membentuk nama file berdasarkan stem input.
            filename = f"{record.stem}{suffix}"

        # Menentukan path tujuan penyimpanan file input.
        destination = input_dir / filename

        # Mengembalikan pointer file ke awal agar seluruh isi file tersalin dari awal.
        upload.file.seek(0)

        # Membuka file tujuan dalam mode write binary.
        with destination.open("wb") as buffer:
            # Menyalin isi file upload ke file tujuan pada folder storage.
            shutil.copyfileobj(upload.file, buffer)

        # Menyimpan path file berdasarkan stem agar proses berikutnya dapat mengambil H-6 sampai H0.
        saved_paths[record.stem] = destination

    # Mengembalikan seluruh path file input yang sudah tersimpan.
    return saved_paths


# Fungsi untuk menulis metadata prediksi ke file JSON.
def write_json(path: Path, payload: dict) -> None:
    # Membuka file tujuan dalam mode tulis dengan encoding UTF-8.
    with path.open("w", encoding="utf-8") as f:
        # Menyimpan dictionary metadata menjadi JSON yang rapi dan tetap mendukung karakter Indonesia.
        json.dump(payload, f, indent=2, ensure_ascii=False)


# Fungsi untuk membaca metadata prediksi dari file JSON.
def read_json(path: Path) -> dict:
    # Membuka file JSON dalam mode baca dengan encoding UTF-8.
    with path.open("r", encoding="utf-8") as f:
        # Mengubah isi file JSON menjadi dictionary Python.
        return json.load(f)


# Fungsi untuk mengubah path file storage menjadi URL publik yang dapat diakses dashboard.
def to_public_storage_url(path: Path) -> str:
    # Mengambil path relatif file terhadap folder storage utama.
    relative = path.relative_to(settings.STORAGE_DIR).as_posix()

    # Membentuk URL publik sesuai mount FastAPI pada /storage.
    return f"/storage/{relative}"
