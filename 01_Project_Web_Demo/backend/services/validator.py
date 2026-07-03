"""Komentar file skripsi:
Validasi input web agar 7 file H-6 sampai H0 sesuai format, ukuran, nama, dan urutan tanggal harian.

Konteks laporan: file ini mendukung BAB IV pada bagian data preparation dan implementasi sistem,
karena memastikan data citra yang masuk ke model ConvLSTM sudah lengkap, valid, dan berurutan.
"""

# Modul os digunakan untuk membaca ukuran file upload dengan proses seek pada file stream.
import os

# dataclass digunakan untuk membuat struktur data hasil validasi upload secara ringkas.
from dataclasses import dataclass

# date digunakan untuk menyimpan tanggal akuisisi citra, sedangkan timedelta untuk validasi urutan harian.
from datetime import date, timedelta

# Path digunakan untuk membaca nama file, stem, dan ekstensi file upload.
from pathlib import Path

# HTTPException, UploadFile, dan status digunakan untuk validasi input serta response error FastAPI.
from fastapi import HTTPException, UploadFile, status


# Dataclass dibuat immutable agar hasil validasi file tidak berubah setelah dibentuk.
@dataclass(frozen=True)
class ValidatedUpload:
    # Stem menyimpan indeks waktu citra, misalnya H-6, H-5, sampai H0.
    stem: str

    # Upload menyimpan object file asli dari FastAPI.
    upload: UploadFile

    # original_filename menyimpan nama file asli yang diunggah pengguna.
    original_filename: str

    # acquired_date menyimpan tanggal citra jika nama file memakai format H-6_YYYY-MM-DD.png.
    acquired_date: date | None = None


# Fungsi internal untuk menormalisasi nama stem agar format input lebih konsisten.
def _normalize_stem(stem: str) -> str:
    # Mengubah stem menjadi huruf besar, menghapus spasi awal/akhir, dan menghapus spasi di tengah.
    clean = stem.upper().strip().replace(" ", "")

    # Menyamakan variasi H-0 menjadi H0 agar tidak dianggap nama file yang berbeda.
    if clean == "H-0":
        # Mengembalikan format standar untuk citra hari terakhir.
        return "H0"

    # Mengembalikan stem yang sudah dinormalisasi.
    return clean


# Fungsi internal untuk menghitung ukuran file upload dalam byte.
def _file_size(upload: UploadFile) -> int:
    # Blok try digunakan karena stream file upload bisa gagal dibaca pada kondisi tertentu.
    try:
        # Menyimpan posisi pointer file saat ini agar dapat dikembalikan setelah pengecekan ukuran.
        current = upload.file.tell()

        # Memindahkan pointer ke akhir file untuk mengetahui total ukuran file.
        upload.file.seek(0, os.SEEK_END)

        # Membaca posisi pointer akhir sebagai ukuran file dalam byte.
        size = upload.file.tell()

        # Mengembalikan pointer file ke posisi semula agar proses penyimpanan file tidak terganggu.
        upload.file.seek(current)

        # Mengembalikan ukuran file.
        return size

    # Jika pengecekan ukuran gagal, file dianggap tidak dapat dibaca.
    except Exception:
        # Mengembalikan 0 agar validasi berikutnya menolak file tersebut.
        return 0


# Fungsi internal untuk membaca identitas file dari nama file.
def _parse_file_identity(filename: str) -> tuple[str, date | None]:
    # Mengambil nama file tanpa ekstensi, menghapus spasi, lalu digunakan sebagai dasar parsing.
    base_name = Path(filename).stem.strip().replace(" ", "")

    # Memisahkan nama file berdasarkan underscore untuk mendukung format H-6_YYYY-MM-DD.png.
    parts = base_name.split("_")

    # Format nama hanya boleh H-6.png atau H-6_YYYY-MM-DD.png, sehingga maksimal 2 bagian.
    if len(parts) > 2:
        # Mengembalikan error jika format nama file terlalu banyak bagian.
        raise HTTPException(
            # Status 400 menunjukkan kesalahan berasal dari input pengguna.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Pesan menjelaskan pola nama file yang benar.
            detail=(
                f"Nama file '{filename}' tidak valid. Gunakan pola H-6.png "
                "atau H-6_YYYY-MM-DD.png."
            ),
        )

    # Bagian pertama nama file digunakan sebagai stem H-6 sampai H0.
    stem = _normalize_stem(parts[0])

    # Tanggal default None jika nama file tidak mencantumkan tanggal.
    acquired_date = None

    # Jika ada bagian kedua, bagian tersebut harus berupa tanggal ISO YYYY-MM-DD.
    if len(parts) == 2:
        # Blok try digunakan untuk menangkap kesalahan format tanggal.
        try:
            # Mengubah teks tanggal menjadi object date.
            acquired_date = date.fromisoformat(parts[1])

        # Jika tanggal tidak sesuai format ISO, validasi gagal.
        except ValueError as exc:
            # Mengembalikan error bahwa tanggal harus memakai format YYYY-MM-DD.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tanggal pada file '{filename}' harus memakai format YYYY-MM-DD.",
            ) from exc

    # Mengembalikan stem dan tanggal akuisisi jika tersedia.
    return stem, acquired_date


# Fungsi internal untuk memastikan tanggal H-6 sampai H0 berurutan harian.
def _validate_daily_dates(records: dict[str, ValidatedUpload], required_stems: list[str]) -> None:
    # Mengambil semua file yang memiliki tanggal pada nama file.
    dated_records = [record for record in records.values() if record.acquired_date is not None]

    # Jika tidak ada file bertanggal, validasi tanggal dilewati agar format lama tetap didukung.
    if not dated_records:
        # Tidak melakukan apa pun karena input tanpa tanggal masih valid.
        return

    # Jika sebagian file memakai tanggal, semua file wajib memakai tanggal.
    if len(dated_records) != len(required_stems):
        # Mengembalikan error karena format bertanggal harus konsisten untuk 7 file.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Jika tanggal dicantumkan pada nama file, semua file H-6 sampai H0 "
                "harus memakai tanggal."
            ),
        )

    # Tanggal awal harus berasal dari file H-6.
    start_date = records[required_stems[0]].acquired_date

    # Jika tanggal H-6 tidak ada, validasi tanggal tidak dapat dilakukan.
    if start_date is None:
        # Mengembalikan error karena H-6 menjadi dasar perhitungan urutan harian.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tanggal H-6 wajib tersedia jika format bertanggal digunakan.",
        )

    # Melakukan pengecekan tanggal untuk setiap stem sesuai urutan H-6 sampai H0.
    for offset, stem in enumerate(required_stems):
        # Tanggal yang diharapkan adalah tanggal H-6 ditambah offset hari.
        expected_date = start_date + timedelta(days=offset)

        # Mengambil tanggal aktual dari file pada stem terkait.
        actual_date = records[stem].acquired_date

        # Membandingkan tanggal aktual dengan tanggal yang seharusnya.
        if actual_date != expected_date:
            # Mengembalikan error jika tanggal tidak berurutan harian.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Tanggal file {stem} harus {expected_date.isoformat()}, "
                    f"tetapi ditemukan {actual_date.isoformat() if actual_date else '-'}."
                ),
            )


# Fungsi utama untuk memvalidasi dan mengurutkan 7 file input sebelum diproses model.
async def validate_and_order_files(
    # Daftar file upload dari form-data endpoint prediksi.
    files: list[UploadFile],
    # Daftar stem wajib, yaitu H-6, H-5, H-4, H-3, H-2, H-1, dan H0.
    required_stems: list[str],
    # Ekstensi file yang diizinkan, misalnya .png, .jpg, dan .jpeg.
    allowed_extensions: set[str],
    # Batas ukuran maksimum file dalam byte.
    max_file_bytes: int,
) -> list[ValidatedUpload]:
    # Memastikan jumlah file sama dengan jumlah stem wajib, yaitu tepat 7 file.
    if len(files) != len(required_stems):
        # Mengembalikan error jika jumlah file tidak tepat.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input harus 7 citra berurutan H-6 s.d. H0.",
        )

    # Dictionary untuk menyimpan hasil validasi berdasarkan stem agar mudah dicek duplikasi dan kelengkapan.
    stem_to_record: dict[str, ValidatedUpload] = {}

    # Melakukan validasi pada setiap file upload.
    for upload in files:
        # Memastikan setiap file memiliki nama file.
        if not upload.filename:
            # Mengembalikan error jika ada file tanpa nama.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Semua file harus memiliki nama.",
            )

        # Mengambil ekstensi file dan mengubahnya ke huruf kecil.
        suffix = Path(upload.filename).suffix.lower()

        # Memastikan ekstensi file termasuk daftar yang diperbolehkan.
        if suffix not in allowed_extensions:
            # Mengembalikan error jika ekstensi bukan PNG/JPG/JPEG.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Format file tidak valid untuk '{upload.filename}'. Gunakan PNG/JPG/JPEG.",
            )

        # Mengambil MIME type file dari metadata upload.
        content_type = (upload.content_type or "").lower()

        # Memastikan file benar-benar dikirim sebagai image/*.
        if not content_type.startswith("image/"):
            # Mengembalikan error jika MIME type bukan gambar.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"MIME type file '{upload.filename}' bukan image/*.",
            )

        # Menghitung ukuran file upload.
        size = _file_size(upload)

        # Menolak file kosong atau file yang tidak bisa dibaca.
        if size <= 0:
            # Mengembalikan error jika ukuran file tidak valid.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{upload.filename}' kosong atau tidak dapat dibaca.",
            )

        # Menolak file yang melebihi batas ukuran maksimum.
        if size > max_file_bytes:
            # Mengembalikan error jika file terlalu besar.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ukuran file '{upload.filename}' melebihi batas.",
            )

        # Membaca stem dan tanggal dari nama file.
        normalized_stem, acquired_date = _parse_file_identity(upload.filename)

        # Memastikan stem file sesuai pola H-6 sampai H0.
        if normalized_stem not in required_stems:
            # Membuat teks daftar stem yang diharapkan untuk pesan error.
            expected = ", ".join(required_stems)

            # Mengembalikan error jika nama file tidak sesuai indeks waktu yang dibutuhkan model.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Nama file tidak sesuai pola. Gunakan: {expected} "
                    "atau format bertanggal seperti H-6_2025-07-13.png."
                ),
            )

        # Memastikan tidak ada dua file dengan stem yang sama.
        if normalized_stem in stem_to_record:
            # Mengembalikan error jika ditemukan duplikasi, misalnya dua file H-6.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Duplikasi file untuk indeks '{normalized_stem}'.",
            )

        # Menyimpan hasil validasi file ke dictionary berdasarkan stem.
        stem_to_record[normalized_stem] = ValidatedUpload(
            # Menyimpan stem hasil normalisasi.
            stem=normalized_stem,
            # Menyimpan object upload agar dapat disimpan oleh service storage.
            upload=upload,
            # Menyimpan nama file asli untuk dokumentasi dan pelacakan.
            original_filename=upload.filename,
            # Menyimpan tanggal akuisisi jika tersedia.
            acquired_date=acquired_date,
        )

    # Mengecek stem wajib yang belum tersedia setelah semua file diproses.
    missing = [stem for stem in required_stems if stem not in stem_to_record]

    # Jika ada file yang hilang, validasi gagal.
    if missing:
        # Menggabungkan daftar stem yang hilang menjadi teks.
        missing_text = ", ".join(missing)

        # Mengembalikan error kelengkapan file.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Urutan file tidak lengkap. File hilang: {missing_text}.",
        )

    # Memvalidasi tanggal harian jika format nama file bertanggal digunakan.
    _validate_daily_dates(stem_to_record, required_stems)

    # Mengurutkan file sesuai required_stems agar input model selalu H-6 sampai H0.
    ordered = [stem_to_record[stem] for stem in required_stems]

    # Mengembalikan daftar file tervalidasi dan sudah berurutan.
    return ordered
