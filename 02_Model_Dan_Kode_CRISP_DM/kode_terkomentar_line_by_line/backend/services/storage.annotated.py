# File anotasi dari `backend/services/storage.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Manajemen storage prediksi untuk menyimpan input, output, metadata, dan URL file hasil web.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import json
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import shutil
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi import UploadFile

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend import settings


# Membuat langkah kerja bernama `create_prediction_dirs`.
def create_prediction_dirs(prediction_id: str) -> tuple[Path, Path]:
    # Menyimpan nilai ke `input_dir` untuk dipakai pada langkah berikutnya.
    input_dir = settings.INPUTS_DIR / prediction_id
    # Menyimpan nilai ke `output_dir` untuk dipakai pada langkah berikutnya.
    output_dir = settings.OUTPUTS_DIR / prediction_id
    # Menyimpan nilai ke `input_dir.mkdir(parents` untuk dipakai pada langkah berikutnya.
    input_dir.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `output_dir.mkdir(parents` untuk dipakai pada langkah berikutnya.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return input_dir, output_dir


# Membuat langkah kerja `save_upload_files` yang bisa menunggu proses web/backend selesai.
async def save_upload_files(
    # Menjelaskan data `ordered_files` yang disimpan atau dikirim pada bagian ini.
    ordered_files: list[tuple[str, UploadFile]],
    # Menjelaskan data `input_dir` yang disimpan atau dikirim pada bagian ini.
    input_dir: Path,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> dict[str, Path]:
    # Menyimpan nilai ke `saved_paths` untuk dipakai pada langkah berikutnya.
    saved_paths: dict[str, Path] = {}
    # Mengulang proses untuk setiap data dalam daftar.
    for stem, upload in ordered_files:
        # Menyimpan nilai ke `suffix` untuk dipakai pada langkah berikutnya.
        suffix = Path(upload.filename or "").suffix.lower() or ".png"
        # Menyimpan nilai ke `destination` untuk dipakai pada langkah berikutnya.
        destination = input_dir / f"{stem}{suffix}"
        # Melanjutkan langkah kerja pada bagian kode ini.
        upload.file.seek(0)
        # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
        with destination.open("wb") as buffer:
            # Melanjutkan langkah kerja pada bagian kode ini.
            shutil.copyfileobj(upload.file, buffer)
        # Menyimpan nilai ke `saved_paths[stem]` untuk dipakai pada langkah berikutnya.
        saved_paths[stem] = destination
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return saved_paths


# Membuat langkah kerja bernama `write_json`.
def write_json(path: Path, payload: dict) -> None:
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with path.open("w", encoding="utf-8") as f:
        # Menyimpan nilai ke `json.dump(payload, f, indent` untuk dipakai pada langkah berikutnya.
        json.dump(payload, f, indent=2, ensure_ascii=False)


# Membuat langkah kerja bernama `read_json`.
def read_json(path: Path) -> dict:
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with path.open("r", encoding="utf-8") as f:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return json.load(f)


# Membuat langkah kerja bernama `to_public_storage_url`.
def to_public_storage_url(path: Path) -> str:
    # Menyimpan nilai ke `relative` untuk dipakai pada langkah berikutnya.
    relative = path.relative_to(settings.STORAGE_DIR).as_posix()
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return f"/storage/{relative}"

