
"""Komentar file skripsi:
Tool pendamping evaluasi geospasial untuk mengunduh data atribut dan vektor FIRMS MODIS_SP.

The script is intentionally separate from the training and web pipelines.
It downloads point attributes in chunks of at most five days, merges them,
and writes one CSV plus a JSON metadata file.

Konteks laporan: file ini mendukung tahapan Evaluation dan Deployment
pada BAB IV sebagai data pembanding geospasial berbasis FIRMS mentah.
"""

from __future__ import annotations

# argparse dipakai agar parameter dataset, patch, threshold, dan output bisa diatur dari command line.
import argparse
# csv dipakai untuk membaca titik FIRMS mentah saat validasi geospasial.
import csv
# json dipakai untuk menyimpan ringkasan eksperimen, metrik, dan metadata proses.
import json
# os dipakai untuk membaca environment runtime seperti mode Colab/WSL atau path sistem.
import os
import time
# datetime dipakai untuk menjaga urutan kronologis citra dan menghitung target H+1.
from datetime import date, datetime, timedelta, timezone
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
EXPECTED_COLUMNS = {"latitude", "longitude", "acq_date"}


# Membaca semua parameter training seperti dataset, patch, threshold, batch, epoch, dan lokasi output.
def parse_args() -> argparse.Namespace:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description="Unduh data titik FIRMS MODIS_SP untuk evaluasi geospasial."
    )
    # Opsi `--start-date` menambah parameter eksekusi script.
    parser.add_argument("--start-date", default="2025-06-21")
    # Opsi `--end-date` menambah parameter eksekusi script.
    parser.add_argument("--end-date", default="2025-11-25")
    # Opsi `--west` menambah parameter eksekusi script.
    parser.add_argument("--west", type=float, default=98.1843)
    # Opsi `--south` menambah parameter eksekusi script.
    parser.add_argument("--south", type=float, default=-1.1808)
    # Opsi `--east` menambah parameter eksekusi script.
    parser.add_argument("--east", type=float, default=106.0157)
    # Opsi `--north` menambah parameter eksekusi script.
    parser.add_argument("--north", type=float, default=2.7799)
    # Opsi `--source` menambah parameter eksekusi script.
    parser.add_argument("--source", default="MODIS_SP")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data-firms-mentah-pendamping/"
            "modis_sp_2025-06-21_2025-11-25_cakupan_peta_project.csv"
        ),
    )
    # Opsi `--request-delay` menambah parameter eksekusi script.
    parser.add_argument("--request-delay", type=float, default=0.25)
    # Hasil ini dikembalikan sebagai output fungsi `parse_args` untuk tahap berikutnya.
    return parser.parse_args()


# Fungsi `date_chunks` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def date_chunks(start: date, end: date, maximum_days: int = 5):
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=maximum_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


# Fungsi `download_text` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def download_text(url: str, retries: int = 3) -> str:
    request = Request(
        url,
        headers={"User-Agent": "sistem-web-skripsi-ta-firms-evaluation/1.0"},
    )
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                # Hasil ini dikembalikan sebagai output fungsi `download_text` untuk tahap berikutnya.
                return response.read().decode("utf-8-sig")
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Gagal mengunduh {url}: {exc}") from exc
            time.sleep(attempt * 2)
    raise RuntimeError(f"Gagal mengunduh {url}")


# Fungsi `parse_csv_response` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def parse_csv_response(content: str, url: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        # Hasil ini dikembalikan sebagai output fungsi `parse_csv_response` untuk tahap berikutnya.
        return [], []

    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames or []
    if not EXPECTED_COLUMNS.issubset(fieldnames):
        preview = content[:300].replace("\n", " ")
        raise RuntimeError(f"Respons FIRMS bukan CSV titik yang valid: {preview} ({url})")
    # Hasil ini dikembalikan sebagai output fungsi `parse_csv_response` untuk tahap berikutnya.
    return fieldnames, list(reader)


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> None:
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parse_args()
    map_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        raise SystemExit(
            "FIRMS_MAP_KEY belum tersedia. Di PowerShell jalankan:\n"
            '$env:FIRMS_MAP_KEY="MAP_KEY_ANDA"'
        )

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise SystemExit("--end-date tidak boleh lebih awal dari --start-date.")

    area = f"{args.west},{args.south},{args.east},{args.north}"
    merged_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    requests_made = 0

    for chunk_start, chunk_end in date_chunks(start, end):
        day_range = (chunk_end - chunk_start).days + 1
        url = (
            f"{API_BASE_URL}/{map_key}/{args.source}/{area}/"
            f"{day_range}/{chunk_start.isoformat()}"
        )
        print(
            "[firms] Mengunduh "
            f"{chunk_start.isoformat()} s.d. {chunk_end.isoformat()}..."
        )
        content = download_text(url)
        chunk_fields, chunk_rows = parse_csv_response(content, url)
        requests_made += 1
        if chunk_fields and not fieldnames:
            fieldnames = chunk_fields
        merged_rows.extend(chunk_rows)
        time.sleep(max(0.0, args.request_delay))

    if not fieldnames:
        fieldnames = [
            "latitude",
            "longitude",
            "brightness",
            "scan",
            "track",
            "acq_date",
            "acq_time",
            "satellite",
            "instrument",
            "confidence",
            "version",
            "bright_t31",
            "frp",
            "daynight",
        ]

    # API chunks do not overlap, but deduplication prevents accidental duplicates.
    unique_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in merged_rows:
        key = tuple(row.get(column, "") for column in fieldnames)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

    dates = sorted({row.get("acq_date", "") for row in unique_rows if row.get("acq_date")})
    metadata = {
        "source": "NASA FIRMS",
        "dataset": args.source,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "bounding_box": {
            "west": args.west,
            "south": args.south,
            "east": args.east,
            "north": args.north,
        },
        "requests_made": requests_made,
        "record_count": len(unique_rows),
        "dates_with_records": len(dates),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_documentation": "https://firms.modaps.eosdis.nasa.gov/api/area/",
    }
    # `metadata_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    metadata_path = args.output.with_suffix(".metadata.json")
    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[firms] Selesai: {len(unique_rows)} titik")
    print(f"[firms] CSV: {args.output.resolve()}")
    print(f"[firms] Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
