# File anotasi dari `backend/services/validator.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Validasi input web agar 7 file H-6 sampai H0 sesuai format, ukuran, dan urutan kebutuhan model.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import os
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi import HTTPException, UploadFile, status


# Membuat langkah kerja bernama `_normalize_stem`.
def _normalize_stem(stem: str) -> str:
    # Menyimpan nilai ke `clean` untuk dipakai pada langkah berikutnya.
    clean = stem.upper().strip().replace(" ", "")
    # Mengecek syarat sebelum melanjutkan proses.
    if clean == "H-0":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "H0"
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return clean


# Membuat langkah kerja bernama `_file_size`.
def _file_size(upload: UploadFile) -> int:
    # Mencoba menjalankan proses yang mungkin gagal.
    try:
        # Menyimpan nilai ke `current` untuk dipakai pada langkah berikutnya.
        current = upload.file.tell()
        # Melanjutkan langkah kerja pada bagian kode ini.
        upload.file.seek(0, os.SEEK_END)
        # Menyimpan nilai ke `size` untuk dipakai pada langkah berikutnya.
        size = upload.file.tell()
        # Melanjutkan langkah kerja pada bagian kode ini.
        upload.file.seek(current)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return size
    # Menangani kesalahan agar program tidak langsung berhenti.
    except Exception:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return 0


# Membuat langkah kerja `validate_and_order_files` yang bisa menunggu proses web/backend selesai.
async def validate_and_order_files(
    # Menjelaskan data `files` yang disimpan atau dikirim pada bagian ini.
    files: list[UploadFile],
    # Menjelaskan data `required_stems` yang disimpan atau dikirim pada bagian ini.
    required_stems: list[str],
    # Menjelaskan data `allowed_extensions` yang disimpan atau dikirim pada bagian ini.
    allowed_extensions: set[str],
    # Menjelaskan data `max_file_bytes` yang disimpan atau dikirim pada bagian ini.
    max_file_bytes: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> list[tuple[str, UploadFile]]:
    # Mengecek syarat sebelum melanjutkan proses.
    if len(files) != len(required_stems):
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise HTTPException(
            # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
            detail="Input harus 7 citra berurutan H-6 s.d. H0.",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Menyimpan nilai ke `stem_to_file` untuk dipakai pada langkah berikutnya.
    stem_to_file: dict[str, UploadFile] = {}
    # Mengulang proses untuk setiap data dalam daftar.
    for upload in files:
        # Mengecek syarat sebelum melanjutkan proses.
        if not upload.filename:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise HTTPException(
                # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
                status_code=status.HTTP_400_BAD_REQUEST,
                # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
                detail="Semua file harus memiliki nama.",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )

        # Menyimpan nilai ke `suffix` untuk dipakai pada langkah berikutnya.
        suffix = Path(upload.filename).suffix.lower()
        # Mengecek syarat sebelum melanjutkan proses.
        if suffix not in allowed_extensions:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise HTTPException(
                # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
                status_code=status.HTTP_400_BAD_REQUEST,
                # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
                detail=f"Format file tidak valid untuk '{upload.filename}'. Gunakan PNG/JPG/JPEG.",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )

        # Menyimpan nilai ke `content_type` untuk dipakai pada langkah berikutnya.
        content_type = (upload.content_type or "").lower()
        # Mengecek syarat sebelum melanjutkan proses.
        if not content_type.startswith("image/"):
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise HTTPException(
                # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
                status_code=status.HTTP_400_BAD_REQUEST,
                # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
                detail=f"MIME type file '{upload.filename}' bukan image/*.",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )

        # Menyimpan nilai ke `size` untuk dipakai pada langkah berikutnya.
        size = _file_size(upload)
        # Mengecek syarat sebelum melanjutkan proses.
        if size <= 0:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise HTTPException(
                # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
                status_code=status.HTTP_400_BAD_REQUEST,
                # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
                detail=f"File '{upload.filename}' kosong atau tidak dapat dibaca.",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
        # Mengecek syarat sebelum melanjutkan proses.
        if size > max_file_bytes:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise HTTPException(
                # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
                status_code=status.HTTP_400_BAD_REQUEST,
                # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
                detail=f"Ukuran file '{upload.filename}' melebihi batas.",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )

        # Menyimpan nilai ke `normalized_stem` untuk dipakai pada langkah berikutnya.
        normalized_stem = _normalize_stem(Path(upload.filename).stem)
        # Mengecek syarat sebelum melanjutkan proses.
        if normalized_stem not in required_stems:
            # Menyimpan nilai ke `expected` untuk dipakai pada langkah berikutnya.
            expected = ", ".join(required_stems)
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise HTTPException(
                # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
                status_code=status.HTTP_400_BAD_REQUEST,
                # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
                detail=f"Nama file tidak sesuai pola. Gunakan: {expected}.",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )

        # Mengecek syarat sebelum melanjutkan proses.
        if normalized_stem in stem_to_file:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise HTTPException(
                # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
                status_code=status.HTTP_400_BAD_REQUEST,
                # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
                detail=f"Duplikasi file untuk indeks '{normalized_stem}'.",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )

        # Menyimpan nilai ke `stem_to_file[normalized_stem]` untuk dipakai pada langkah berikutnya.
        stem_to_file[normalized_stem] = upload

    # Menyimpan nilai ke `missing` untuk dipakai pada langkah berikutnya.
    missing = [stem for stem in required_stems if stem not in stem_to_file]
    # Mengecek syarat sebelum melanjutkan proses.
    if missing:
        # Menyimpan nilai ke `missing_text` untuk dipakai pada langkah berikutnya.
        missing_text = ", ".join(missing)
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise HTTPException(
            # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
            detail=f"Urutan file tidak lengkap. File hilang: {missing_text}.",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Menyimpan nilai ke `ordered` untuk dipakai pada langkah berikutnya.
    ordered = [(stem, stem_to_file[stem]) for stem in required_stems]
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return ordered

