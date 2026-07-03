# File anotasi dari `tools/download_firms_modis_evaluation.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Evaluation, yaitu mengukur hasil prediksi dan validasi.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Tool pendamping evaluasi geospasial untuk mengunduh data atribut dan vektor FIRMS MODIS_SP.

# Isi catatan penjelasan pada bagian kode ini.
The script is intentionally separate from the training and web pipelines.
# Isi catatan penjelasan pada bagian kode ini.
It downloads point attributes in chunks of at most five days, merges them,
# Isi catatan penjelasan pada bagian kode ini.
and writes one CSV plus a JSON metadata file.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan Evaluation dan Deployment
# Isi catatan penjelasan pada bagian kode ini.
pada BAB IV sebagai data pembanding geospasial berbasis FIRMS mentah.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Menjaga cara baca kode Python tetap cocok dengan versi yang digunakan.
from __future__ import annotations

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import argparse
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import csv
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import json
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import os
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import time
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from datetime import date, datetime, timedelta, timezone
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from urllib.error import HTTPError, URLError
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from urllib.request import Request, urlopen


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
API_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
EXPECTED_COLUMNS = {"latitude", "longitude", "acq_date"}


# Membuat langkah kerja bernama `parse_args`.
def parse_args() -> argparse.Namespace:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description="Unduh data titik FIRMS MODIS_SP untuk evaluasi geospasial."
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--start-date", default="2025-06-21")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--end-date", default="2025-11-25")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--west", type=float, default=98.1843)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--south", type=float, default=-1.1808)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--east", type=float, default=106.0157)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--north", type=float, default=2.7799)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--source", default="MODIS_SP")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--output",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=Path,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=Path(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "data-firms-mentah-pendamping/"
            # Melanjutkan langkah kerja pada bagian kode ini.
            "modis_sp_2025-06-21_2025-11-25_cakupan_peta_project.csv"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--request-delay", type=float, default=0.25)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser.parse_args()


# Membuat langkah kerja bernama `date_chunks`.
def date_chunks(start: date, end: date, maximum_days: int = 5):
    # Menyimpan nilai ke `current` untuk dipakai pada langkah berikutnya.
    current = start
    # Mengulang proses selama syaratnya masih terpenuhi.
    while current <= end:
        # Menyimpan nilai ke `chunk_end` untuk dipakai pada langkah berikutnya.
        chunk_end = min(current + timedelta(days=maximum_days - 1), end)
        # Melanjutkan langkah kerja pada bagian kode ini.
        yield current, chunk_end
        # Menyimpan nilai ke `current` untuk dipakai pada langkah berikutnya.
        current = chunk_end + timedelta(days=1)


# Membuat langkah kerja bernama `download_text`.
def download_text(url: str, retries: int = 3) -> str:
    # Menyimpan nilai ke `request` untuk dipakai pada langkah berikutnya.
    request = Request(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        url,
        # Menyimpan nilai ke `headers` untuk dipakai pada langkah berikutnya.
        headers={"User-Agent": "sistem-web-skripsi-ta-firms-evaluation/1.0"},
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengulang proses untuk setiap data dalam daftar.
    for attempt in range(1, retries + 1):
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
            with urlopen(request, timeout=60) as response:
                # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
                return response.read().decode("utf-8-sig")
        # Menangani kesalahan agar program tidak langsung berhenti.
        except (HTTPError, URLError, TimeoutError) as exc:
            # Mengecek syarat sebelum melanjutkan proses.
            if attempt == retries:
                # Menghentikan proses dan memberi pesan kesalahan yang jelas.
                raise RuntimeError(f"Gagal mengunduh {url}: {exc}") from exc
            # Melanjutkan langkah kerja pada bagian kode ini.
            time.sleep(attempt * 2)
    # Menghentikan proses dan memberi pesan kesalahan yang jelas.
    raise RuntimeError(f"Gagal mengunduh {url}")


# Membuat langkah kerja bernama `parse_csv_response`.
def parse_csv_response(content: str, url: str) -> tuple[list[str], list[dict[str, str]]]:
    # Menyimpan nilai ke `lines` untuk dipakai pada langkah berikutnya.
    lines = [line for line in content.splitlines() if line.strip()]
    # Mengecek syarat sebelum melanjutkan proses.
    if not lines:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return [], []

    # Menyimpan nilai ke `reader` untuk dipakai pada langkah berikutnya.
    reader = csv.DictReader(lines)
    # Menyimpan nilai ke `fieldnames` untuk dipakai pada langkah berikutnya.
    fieldnames = reader.fieldnames or []
    # Mengecek syarat sebelum melanjutkan proses.
    if not EXPECTED_COLUMNS.issubset(fieldnames):
        # Menyimpan nilai ke `preview` untuk dipakai pada langkah berikutnya.
        preview = content[:300].replace("\n", " ")
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise RuntimeError(f"Respons FIRMS bukan CSV titik yang valid: {preview} ({url})")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return fieldnames, list(reader)


# Membuat langkah kerja bernama `main`.
def main() -> None:
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parse_args()
    # Menyimpan nilai ke `map_key` untuk dipakai pada langkah berikutnya.
    map_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    # Mengecek syarat sebelum melanjutkan proses.
    if not map_key:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise SystemExit(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "FIRMS_MAP_KEY belum tersedia. Di PowerShell jalankan:\n"
            # Menyimpan nilai ke `'$env` untuk dipakai pada langkah berikutnya.
            '$env:FIRMS_MAP_KEY="MAP_KEY_ANDA"'
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Menyimpan nilai ke `start` untuk dipakai pada langkah berikutnya.
    start = date.fromisoformat(args.start_date)
    # Menyimpan nilai ke `end` untuk dipakai pada langkah berikutnya.
    end = date.fromisoformat(args.end_date)
    # Mengecek syarat sebelum melanjutkan proses.
    if end < start:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise SystemExit("--end-date tidak boleh lebih awal dari --start-date.")

    # Menyimpan nilai ke `area` untuk dipakai pada langkah berikutnya.
    area = f"{args.west},{args.south},{args.east},{args.north}"
    # Menyimpan nilai ke `merged_rows` untuk dipakai pada langkah berikutnya.
    merged_rows: list[dict[str, str]] = []
    # Menyimpan nilai ke `fieldnames` untuk dipakai pada langkah berikutnya.
    fieldnames: list[str] = []
    # Menyimpan nilai ke `requests_made` untuk dipakai pada langkah berikutnya.
    requests_made = 0

    # Mengulang proses untuk setiap data dalam daftar.
    for chunk_start, chunk_end in date_chunks(start, end):
        # Menyimpan nilai ke `day_range` untuk dipakai pada langkah berikutnya.
        day_range = (chunk_end - chunk_start).days + 1
        # Menyimpan nilai ke `url` untuk dipakai pada langkah berikutnya.
        url = (
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{API_BASE_URL}/{map_key}/{args.source}/{area}/"
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{day_range}/{chunk_start.isoformat()}"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "[firms] Mengunduh "
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"{chunk_start.isoformat()} s.d. {chunk_end.isoformat()}..."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `content` untuk dipakai pada langkah berikutnya.
        content = download_text(url)
        # Menyimpan nilai ke `chunk_fields, chunk_rows` untuk dipakai pada langkah berikutnya.
        chunk_fields, chunk_rows = parse_csv_response(content, url)
        # Menyimpan nilai ke `requests_made +` untuk dipakai pada langkah berikutnya.
        requests_made += 1
        # Mengecek syarat sebelum melanjutkan proses.
        if chunk_fields and not fieldnames:
            # Menyimpan nilai ke `fieldnames` untuk dipakai pada langkah berikutnya.
            fieldnames = chunk_fields
        # Melanjutkan langkah kerja pada bagian kode ini.
        merged_rows.extend(chunk_rows)
        # Melanjutkan langkah kerja pada bagian kode ini.
        time.sleep(max(0.0, args.request_delay))

    # Mengecek syarat sebelum melanjutkan proses.
    if not fieldnames:
        # Menyimpan nilai ke `fieldnames` untuk dipakai pada langkah berikutnya.
        fieldnames = [
            # Melanjutkan langkah kerja pada bagian kode ini.
            "latitude",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "longitude",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "brightness",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "scan",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "track",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "acq_date",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "acq_time",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "satellite",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "instrument",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "confidence",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "version",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "bright_t31",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "frp",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "daynight",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]

    # Catatan asli dari pembuat kode.
    # API chunks do not overlap, but deduplication prevents accidental duplicates.
    # Menyimpan nilai ke `unique_rows` untuk dipakai pada langkah berikutnya.
    unique_rows: list[dict[str, str]] = []
    # Menyimpan nilai ke `seen` untuk dipakai pada langkah berikutnya.
    seen: set[tuple[str, ...]] = set()
    # Mengulang proses untuk setiap data dalam daftar.
    for row in merged_rows:
        # Menyimpan nilai ke `key` untuk dipakai pada langkah berikutnya.
        key = tuple(row.get(column, "") for column in fieldnames)
        # Mengecek syarat sebelum melanjutkan proses.
        if key not in seen:
            # Melanjutkan langkah kerja pada bagian kode ini.
            seen.add(key)
            # Melanjutkan langkah kerja pada bagian kode ini.
            unique_rows.append(row)

    # Menyimpan nilai ke `args.output.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        # Menyimpan nilai ke `writer` untuk dipakai pada langkah berikutnya.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Melanjutkan langkah kerja pada bagian kode ini.
        writer.writeheader()
        # Melanjutkan langkah kerja pada bagian kode ini.
        writer.writerows(unique_rows)

    # Menyimpan nilai ke `dates` untuk dipakai pada langkah berikutnya.
    dates = sorted({row.get("acq_date", "") for row in unique_rows if row.get("acq_date")})
    # Menyimpan nilai ke `metadata` untuk dipakai pada langkah berikutnya.
    metadata = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "source": "NASA FIRMS",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dataset": args.source,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "start_date": args.start_date,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "end_date": args.end_date,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "bounding_box": {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "west": args.west,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "south": args.south,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "east": args.east,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "north": args.north,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Melanjutkan langkah kerja pada bagian kode ini.
        "requests_made": requests_made,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "record_count": len(unique_rows),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "dates_with_records": len(dates),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "api_documentation": "https://firms.modaps.eosdis.nasa.gov/api/area/",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Menyimpan nilai ke `metadata_path` untuk dipakai pada langkah berikutnya.
    metadata_path = args.output.with_suffix(".metadata.json")
    # Menyimpan nilai ke `metadata_path.write_text(json.dumps(metadata, indent` untuk dipakai pada langkah berikutnya.
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[firms] Selesai: {len(unique_rows)} titik")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[firms] CSV: {args.output.resolve()}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[firms] Metadata: {metadata_path.resolve()}")


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Melanjutkan langkah kerja pada bagian kode ini.
    main()
