# File anotasi dari `tools/geospatial_validate_firms_prediction.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Evaluation, yaitu mengukur hasil prediksi dan validasi.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Tool evaluasi geospasial untuk membandingkan titik FIRMS aktual dengan area prediksi model.

# Isi catatan penjelasan pada bagian kode ini.
This tool is intentionally separate from training and the web backend. It adds
# Isi catatan penjelasan pada bagian kode ini.
the geospatial evidence requested for the thesis: FIRMS point attributes,
# Isi catatan penjelasan pada bagian kode ini.
optional administrative spatial join, distance-based validation, and overlay
# Isi catatan penjelasan pada bagian kode ini.
figures.

# Isi catatan penjelasan pada bagian kode ini.
It avoids heavy GIS dependencies. Administrative boundaries are accepted as
# Isi catatan penjelasan pada bagian kode ini.
GeoJSON so point-in-polygon can be done with a small local implementation.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan Evaluation pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
khususnya pembuktian data atribut, data vektor, dan validasi jarak spasial.
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
import math
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from collections import defaultdict
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from datetime import datetime, timezone
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from typing import Any

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image, ImageDraw


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_FIRMS_CSV = Path(
    # Melanjutkan langkah kerja pada bagian kode ini.
    "data-firms-mentah-pendamping/"
    # Melanjutkan langkah kerja pada bagian kode ini.
    "modis_sp_2025-06-21_2025-11-25_cakupan_peta_project.csv"
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_DATASET_DIR = Path("Ipynb/Dataset History Fire Hotspot In Riau Province PNG")
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
DEFAULT_OUTPUT_DIR = Path("artifacts/geospatial_firms_validation")

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
WEB_MERCATOR_MAX_LAT = 85.05112878
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
EARTH_EQUATORIAL_CIRCUMFERENCE_M = 40075016.68557849
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
TILE_SIZE = 256.0


# Membuat langkah kerja bernama `parse_args`.
def parse_args() -> argparse.Namespace:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description=(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Evaluasi geospasial titik FIRMS terhadap peta prediksi risiko "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "hotspot H+1."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--firms-csv", type=Path, default=DEFAULT_FIRMS_CSV)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--target-date",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Filter tanggal aktual FIRMS, contoh 2025-07-19. Kosongkan untuk semua tanggal.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--probability-npy",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=Path,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=None,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="File probability_H+1.npy hasil model. Jika kosong, hanya dibuat tabel titik/overlay FIRMS.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--target-image",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=Path,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=None,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="PNG/JPG tanggal target untuk background overlay. Jika kosong, dicari dari dataset-dir.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--admin-geojson",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=Path,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=None,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="GeoJSON batas administrasi kabupaten/kecamatan untuk spatial join.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--threshold", type=float, default=0.55)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--center-lon", type=float, default=102.1)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--center-lat", type=float, default=0.8)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--zoom", type=float, default=8.1)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--image-width", type=int, default=1528)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--image-height", type=int, default=773)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--distance-km",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="1,3,5",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Daftar radius evaluasi dalam kilometer, contoh: 1,3,5.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--kabupaten-fields",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="kabupaten_kota,shapeName,NAME_2,WADMKK,KABUPATEN,KAB_KOTA,KABUPATEN_KOTA,KABKOT,NAMOBJ",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Prioritas nama field kabupaten/kota pada GeoJSON.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--kecamatan-fields",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="NAME_3,WADMKC,KECAMATAN,KEC,NAMOBJ",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Prioritas nama field kecamatan pada GeoJSON.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--output-dir",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=Path,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=DEFAULT_OUTPUT_DIR,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--max-boundary-features",
        # Menyimpan nilai ke `type` untuk dipakai pada langkah berikutnya.
        type=int,
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default=0,
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Batas jumlah feature admin yang dibaca. 0 berarti semua.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser.parse_args()


# Membuat langkah kerja bernama `split_fields`.
def split_fields(value: str) -> list[str]:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return [part.strip() for part in value.split(",") if part.strip()]


# Membuat langkah kerja bernama `split_distances`.
def split_distances(value: str) -> list[float]:
    # Menyimpan nilai ke `distances` untuk dipakai pada langkah berikutnya.
    distances = [float(part.strip()) for part in value.split(",") if part.strip()]
    # Mengecek syarat sebelum melanjutkan proses.
    if not distances:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise SystemExit("--distance-km minimal berisi satu nilai.")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return sorted(distances)


# Membuat langkah kerja bernama `web_mercator_global_pixel`.
def web_mercator_global_pixel(lon: float, lat: float, zoom: float) -> tuple[float, float]:
    # Menyimpan nilai ke `lat` untuk dipakai pada langkah berikutnya.
    lat = max(min(lat, WEB_MERCATOR_MAX_LAT), -WEB_MERCATOR_MAX_LAT)
    # Menyimpan nilai ke `scale` untuk dipakai pada langkah berikutnya.
    scale = TILE_SIZE * (2.0**zoom)
    # Menyimpan nilai ke `x` untuk dipakai pada langkah berikutnya.
    x = (lon + 180.0) / 360.0 * scale
    # Menyimpan nilai ke `sin_lat` untuk dipakai pada langkah berikutnya.
    sin_lat = math.sin(math.radians(lat))
    # Menyimpan nilai ke `y` untuk dipakai pada langkah berikutnya.
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return x, y


# Membuat langkah kerja bernama `lonlat_to_pixel`.
def lonlat_to_pixel(
    # Menjelaskan data `lon` yang disimpan atau dikirim pada bagian ini.
    lon: float,
    # Menjelaskan data `lat` yang disimpan atau dikirim pada bagian ini.
    lat: float,
    # Melanjutkan langkah kerja pada bagian kode ini.
    *,
    # Menjelaskan data `center_lon` yang disimpan atau dikirim pada bagian ini.
    center_lon: float,
    # Menjelaskan data `center_lat` yang disimpan atau dikirim pada bagian ini.
    center_lat: float,
    # Menjelaskan data `zoom` yang disimpan atau dikirim pada bagian ini.
    zoom: float,
    # Menjelaskan data `image_width` yang disimpan atau dikirim pada bagian ini.
    image_width: int,
    # Menjelaskan data `image_height` yang disimpan atau dikirim pada bagian ini.
    image_height: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> tuple[float, float]:
    # Menyimpan nilai ke `center_x, center_y` untuk dipakai pada langkah berikutnya.
    center_x, center_y = web_mercator_global_pixel(center_lon, center_lat, zoom)
    # Menyimpan nilai ke `x, y` untuk dipakai pada langkah berikutnya.
    x, y = web_mercator_global_pixel(lon, lat, zoom)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return x - center_x + image_width / 2.0, y - center_y + image_height / 2.0


# Membuat langkah kerja bernama `km_per_pixel`.
def km_per_pixel(center_lat: float, zoom: float) -> float:
    # Menyimpan nilai ke `meters` untuk dipakai pada langkah berikutnya.
    meters = (
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        EARTH_EQUATORIAL_CIRCUMFERENCE_M
        # Melanjutkan langkah kerja pada bagian kode ini.
        * math.cos(math.radians(center_lat))
        # Melanjutkan langkah kerja pada bagian kode ini.
        / (TILE_SIZE * (2.0**zoom))
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return meters / 1000.0


# Membuat langkah kerja bernama `read_firms_rows`.
def read_firms_rows(path: Path, target_date: str) -> tuple[list[str], list[dict[str, str]]]:
    # Mengecek syarat sebelum melanjutkan proses.
    if not path.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError(f"CSV FIRMS tidak ditemukan: {path}")
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        # Menyimpan nilai ke `reader` untuk dipakai pada langkah berikutnya.
        reader = csv.DictReader(handle)
        # Menyimpan nilai ke `fieldnames` untuk dipakai pada langkah berikutnya.
        fieldnames = reader.fieldnames or []
        # Menyimpan nilai ke `required` untuk dipakai pada langkah berikutnya.
        required = {"latitude", "longitude", "acq_date"}
        # Menyimpan nilai ke `missing` untuk dipakai pada langkah berikutnya.
        missing = required.difference(fieldnames)
        # Mengecek syarat sebelum melanjutkan proses.
        if missing:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise ValueError(f"CSV FIRMS tidak memiliki kolom wajib: {sorted(missing)}")
        # Menyimpan nilai ke `rows` untuk dipakai pada langkah berikutnya.
        rows = []
        # Mengulang proses untuk setiap data dalam daftar.
        for row in reader:
            # Mengecek syarat sebelum melanjutkan proses.
            if target_date and row.get("acq_date") != target_date:
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Melanjutkan langkah kerja pada bagian kode ini.
            rows.append(row)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return fieldnames, rows


# Membuat langkah kerja bernama `load_probability`.
def load_probability(path: Path | None) -> np.ndarray | None:
    # Mengecek syarat sebelum melanjutkan proses.
    if path is None:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return None
    # Mengecek syarat sebelum melanjutkan proses.
    if not path.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError(f"Probability NPY tidak ditemukan: {path}")
    # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
    probability = np.load(path).astype(np.float32)
    # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
    probability = np.squeeze(probability)
    # Mengecek syarat sebelum melanjutkan proses.
    if probability.ndim != 2:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError(
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"Probability NPY harus menjadi array 2D setelah squeeze, didapat {probability.shape}"
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.clip(probability, 0.0, 1.0)


# Membuat langkah kerja bernama `resize_probability`.
def resize_probability(probability: np.ndarray, width: int, height: int) -> np.ndarray:
    # Mengecek syarat sebelum melanjutkan proses.
    if probability.shape == (height, width):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return probability
    # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
    image = Image.fromarray((np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8))
    # Menyimpan nilai ke `resized` untuk dipakai pada langkah berikutnya.
    resized = image.resize((width, height), Image.BILINEAR)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.asarray(resized, dtype=np.float32) / 255.0


# Membuat langkah kerja bernama `find_target_image`.
def find_target_image(dataset_dir: Path, target_date: str) -> Path | None:
    # Mengecek syarat sebelum melanjutkan proses.
    if not target_date or not dataset_dir.exists():
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return None
    # Menyimpan nilai ke `patterns` untuk dipakai pada langkah berikutnya.
    patterns = [
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"FIRMS_{target_date}*.png",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"FIRMS_{target_date}*.jpg",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"FIRMS_{target_date}*.jpeg",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ]
    # Mengulang proses untuk setiap data dalam daftar.
    for pattern in patterns:
        # Menyimpan nilai ke `matches` untuk dipakai pada langkah berikutnya.
        matches = sorted(dataset_dir.glob(pattern))
        # Mengecek syarat sebelum melanjutkan proses.
        if matches:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return matches[0]
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return None


# Membuat langkah kerja bernama `point_in_ring`.
def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    # Menyimpan nilai ke `inside` untuk dipakai pada langkah berikutnya.
    inside = False
    # Mengecek syarat sebelum melanjutkan proses.
    if len(ring) < 3:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return False
    # Menyimpan nilai ke `x, y` untuk dipakai pada langkah berikutnya.
    x, y = lon, lat
    # Menyimpan nilai ke `previous_x, previous_y` untuk dipakai pada langkah berikutnya.
    previous_x, previous_y = ring[-1][0], ring[-1][1]
    # Mengulang proses untuk setiap data dalam daftar.
    for current in ring:
        # Menyimpan nilai ke `current_x, current_y` untuk dipakai pada langkah berikutnya.
        current_x, current_y = current[0], current[1]
        # Menyimpan nilai ke `crosses` untuk dipakai pada langkah berikutnya.
        crosses = (current_y > y) != (previous_y > y)
        # Mengecek syarat sebelum melanjutkan proses.
        if crosses:
            # Menyimpan nilai ke `denom` untuk dipakai pada langkah berikutnya.
            denom = previous_y - current_y
            # Mengecek syarat sebelum melanjutkan proses.
            if abs(denom) < 1e-12:
                # Menyimpan nilai ke `intersection_x` untuk dipakai pada langkah berikutnya.
                intersection_x = current_x
            # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
            else:
                # Menyimpan nilai ke `intersection_x` untuk dipakai pada langkah berikutnya.
                intersection_x = (
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    (previous_x - current_x) * (y - current_y) / denom + current_x
                # Menutup susunan data atau perintah yang dimulai sebelumnya.
                )
            # Mengecek syarat sebelum melanjutkan proses.
            if x < intersection_x:
                # Menyimpan nilai ke `inside` untuk dipakai pada langkah berikutnya.
                inside = not inside
        # Menyimpan nilai ke `previous_x, previous_y` untuk dipakai pada langkah berikutnya.
        previous_x, previous_y = current_x, current_y
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return inside


# Membuat langkah kerja bernama `point_in_polygon`.
def point_in_polygon(lon: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    # Mengecek syarat sebelum melanjutkan proses.
    if not polygon:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return False
    # Mengecek syarat sebelum melanjutkan proses.
    if not point_in_ring(lon, lat, polygon[0]):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return False
    # Catatan asli dari pembuat kode.
    # Holes exclude the point.
    # Mengulang proses untuk setiap data dalam daftar.
    for hole in polygon[1:]:
        # Mengecek syarat sebelum melanjutkan proses.
        if point_in_ring(lon, lat, hole):
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return False
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return True


# Membuat langkah kerja bernama `geometry_contains_point`.
def geometry_contains_point(geometry: dict[str, Any], lon: float, lat: float) -> bool:
    # Menyimpan nilai ke `geom_type` untuk dipakai pada langkah berikutnya.
    geom_type = geometry.get("type")
    # Menyimpan nilai ke `coordinates` untuk dipakai pada langkah berikutnya.
    coordinates = geometry.get("coordinates", [])
    # Mengecek syarat sebelum melanjutkan proses.
    if geom_type == "Polygon":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return point_in_polygon(lon, lat, coordinates)
    # Mengecek syarat sebelum melanjutkan proses.
    if geom_type == "MultiPolygon":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return any(point_in_polygon(lon, lat, polygon) for polygon in coordinates)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return False


# Membuat langkah kerja bernama `geometry_pixel_paths`.
def geometry_pixel_paths(
    # Menjelaskan data `geometry` yang disimpan atau dikirim pada bagian ini.
    geometry: dict[str, Any],
    # Melanjutkan langkah kerja pada bagian kode ini.
    *,
    # Menjelaskan data `center_lon` yang disimpan atau dikirim pada bagian ini.
    center_lon: float,
    # Menjelaskan data `center_lat` yang disimpan atau dikirim pada bagian ini.
    center_lat: float,
    # Menjelaskan data `zoom` yang disimpan atau dikirim pada bagian ini.
    zoom: float,
    # Menjelaskan data `image_width` yang disimpan atau dikirim pada bagian ini.
    image_width: int,
    # Menjelaskan data `image_height` yang disimpan atau dikirim pada bagian ini.
    image_height: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> list[list[tuple[float, float]]]:
    # Menyimpan nilai ke `paths` untuk dipakai pada langkah berikutnya.
    paths: list[list[tuple[float, float]]] = []
    # Menyimpan nilai ke `geom_type` untuk dipakai pada langkah berikutnya.
    geom_type = geometry.get("type")
    # Menyimpan nilai ke `coordinates` untuk dipakai pada langkah berikutnya.
    coordinates = geometry.get("coordinates", [])
    # Menyimpan nilai ke `polygons` untuk dipakai pada langkah berikutnya.
    polygons = [coordinates] if geom_type == "Polygon" else coordinates if geom_type == "MultiPolygon" else []
    # Mengulang proses untuk setiap data dalam daftar.
    for polygon in polygons:
        # Mengecek syarat sebelum melanjutkan proses.
        if not polygon:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Menyimpan nilai ke `outer_ring` untuk dipakai pada langkah berikutnya.
        outer_ring = polygon[0]
        # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
        path = [
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            lonlat_to_pixel(
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                lon,
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                lat,
                # Menyimpan nilai ke `center_lon` untuk dipakai pada langkah berikutnya.
                center_lon=center_lon,
                # Menyimpan nilai ke `center_lat` untuk dipakai pada langkah berikutnya.
                center_lat=center_lat,
                # Menyimpan nilai ke `zoom` untuk dipakai pada langkah berikutnya.
                zoom=zoom,
                # Menyimpan nilai ke `image_width` untuk dipakai pada langkah berikutnya.
                image_width=image_width,
                # Menyimpan nilai ke `image_height` untuk dipakai pada langkah berikutnya.
                image_height=image_height,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
            # Mengulang proses untuk setiap data dalam daftar.
            for lon, lat, *_ in outer_ring
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]
        # Melanjutkan langkah kerja pada bagian kode ini.
        paths.append(path)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return paths


# Membuat langkah kerja bernama `pick_property`.
def pick_property(properties: dict[str, Any], fields: list[str]) -> str:
    # Menyimpan nilai ke `normalized` untuk dipakai pada langkah berikutnya.
    normalized = {str(key).upper(): value for key, value in properties.items()}
    # Mengulang proses untuk setiap data dalam daftar.
    for field in fields:
        # Menyimpan nilai ke `value` untuk dipakai pada langkah berikutnya.
        value = normalized.get(field.upper())
        # Mengecek syarat sebelum melanjutkan proses.
        if value not in (None, ""):
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return str(value)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return ""


# Membuat langkah kerja bernama `load_admin_features`.
def load_admin_features(
    # Menjelaskan data `path` yang disimpan atau dikirim pada bagian ini.
    path: Path | None,
    # Menjelaskan data `kabupaten_fields` yang disimpan atau dikirim pada bagian ini.
    kabupaten_fields: list[str],
    # Menjelaskan data `kecamatan_fields` yang disimpan atau dikirim pada bagian ini.
    kecamatan_fields: list[str],
    # Menjelaskan data `max_features` yang disimpan atau dikirim pada bagian ini.
    max_features: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> list[dict[str, Any]]:
    # Mengecek syarat sebelum melanjutkan proses.
    if path is None:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return []
    # Mengecek syarat sebelum melanjutkan proses.
    if not path.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise FileNotFoundError(f"GeoJSON administrasi tidak ditemukan: {path}")
    # Menyimpan nilai ke `data` untuk dipakai pada langkah berikutnya.
    data = json.loads(path.read_text(encoding="utf-8"))
    # Mengecek syarat sebelum melanjutkan proses.
    if data.get("type") != "FeatureCollection":
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError("GeoJSON administrasi harus bertipe FeatureCollection.")
    # Menyimpan nilai ke `features` untuk dipakai pada langkah berikutnya.
    features = []
    # Mengulang proses untuk setiap data dalam daftar.
    for feature in data.get("features", []):
        # Menyimpan nilai ke `properties` untuk dipakai pada langkah berikutnya.
        properties = feature.get("properties", {}) or {}
        # Menyimpan nilai ke `geometry` untuk dipakai pada langkah berikutnya.
        geometry = feature.get("geometry") or {}
        # Mengecek syarat sebelum melanjutkan proses.
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        features.append(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                # Melanjutkan langkah kerja pada bagian kode ini.
                "geometry": geometry,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "properties": properties,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "kabupaten": pick_property(properties, kabupaten_fields),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "kecamatan": pick_property(properties, kecamatan_fields),
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            }
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Mengecek syarat sebelum melanjutkan proses.
        if max_features and len(features) >= max_features:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            break
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return features


# Membuat langkah kerja bernama `spatial_join`.
def spatial_join(lon: float, lat: float, features: list[dict[str, Any]]) -> tuple[str, str]:
    # Mengulang proses untuk setiap data dalam daftar.
    for feature in features:
        # Mengecek syarat sebelum melanjutkan proses.
        if geometry_contains_point(feature["geometry"], lon, lat):
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return feature.get("kabupaten", ""), feature.get("kecamatan", "")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "", ""


# Membuat langkah kerja bernama `nearest_positive_distance_px`.
def nearest_positive_distance_px(
    # Menjelaskan data `mask` yang disimpan atau dikirim pada bagian ini.
    mask: np.ndarray,
    # Menjelaskan data `x` yang disimpan atau dikirim pada bagian ini.
    x: float,
    # Menjelaskan data `y` yang disimpan atau dikirim pada bagian ini.
    y: float,
    # Menjelaskan data `max_radius_px` yang disimpan atau dikirim pada bagian ini.
    max_radius_px: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> float | None:
    # Menyimpan nilai ke `height, width` untuk dipakai pada langkah berikutnya.
    height, width = mask.shape
    # Menyimpan nilai ke `center_x` untuk dipakai pada langkah berikutnya.
    center_x = int(round(x))
    # Menyimpan nilai ke `center_y` untuk dipakai pada langkah berikutnya.
    center_y = int(round(y))
    # Menyimpan nilai ke `left` untuk dipakai pada langkah berikutnya.
    left = max(0, center_x - max_radius_px)
    # Menyimpan nilai ke `right` untuk dipakai pada langkah berikutnya.
    right = min(width - 1, center_x + max_radius_px)
    # Menyimpan nilai ke `top` untuk dipakai pada langkah berikutnya.
    top = max(0, center_y - max_radius_px)
    # Menyimpan nilai ke `bottom` untuk dipakai pada langkah berikutnya.
    bottom = min(height - 1, center_y + max_radius_px)
    # Mengecek syarat sebelum melanjutkan proses.
    if left > right or top > bottom:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return None
    # Menyimpan nilai ke `window` untuk dipakai pada langkah berikutnya.
    window = mask[top : bottom + 1, left : right + 1]
    # Menyimpan nilai ke `points` untuk dipakai pada langkah berikutnya.
    points = np.argwhere(window)
    # Mengecek syarat sebelum melanjutkan proses.
    if points.size == 0:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return None
    # Catatan asli dari pembuat kode.
    # np.argwhere returns y, x positions in the local window.
    # Menyimpan nilai ke `local_y` untuk dipakai pada langkah berikutnya.
    local_y = points[:, 0] + top
    # Menyimpan nilai ke `local_x` untuk dipakai pada langkah berikutnya.
    local_x = points[:, 1] + left
    # Menyimpan nilai ke `distances` untuk dipakai pada langkah berikutnya.
    distances = np.sqrt((local_x - x) ** 2 + (local_y - y) ** 2)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return float(np.min(distances))


# Membuat langkah kerja bernama `write_csv_safe`.
def write_csv_safe(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    # Menyimpan nilai ke `path.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with path.open("w", newline="", encoding="utf-8") as handle:
        # Menyimpan nilai ke `writer` untuk dipakai pada langkah berikutnya.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Melanjutkan langkah kerja pada bagian kode ini.
        writer.writeheader()
        # Melanjutkan langkah kerja pada bagian kode ini.
        writer.writerows(rows)


# Membuat langkah kerja bernama `summarize_by`.
def summarize_by(rows: list[dict[str, Any]], group_field: str, distances: list[float]) -> list[dict[str, Any]]:
    # Menyimpan nilai ke `grouped` untuk dipakai pada langkah berikutnya.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Mengulang proses untuk setiap data dalam daftar.
    for row in rows:
        # Menyimpan nilai ke `key` untuk dipakai pada langkah berikutnya.
        key = str(row.get(group_field) or "Tidak teridentifikasi")
        # Melanjutkan langkah kerja pada bagian kode ini.
        grouped[key].append(row)
    # Menyimpan nilai ke `summary_rows` untuk dipakai pada langkah berikutnya.
    summary_rows = []
    # Mengulang proses untuk setiap data dalam daftar.
    for key, items in sorted(grouped.items()):
        # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
        summary: dict[str, Any] = {group_field: key, "jumlah_titik": len(items)}
        # Mengulang proses untuk setiap data dalam daftar.
        for distance in distances:
            # Menyimpan nilai ke `field` untuk dipakai pada langkah berikutnya.
            field = f"match_{format_distance(distance)}km"
            # Menyimpan nilai ke `count` untuk dipakai pada langkah berikutnya.
            count = sum(1 for item in items if str(item.get(field, "")).lower() == "true")
            # Menyimpan nilai ke `summary[f"sesuai_{format_distance(distance)}km"]` untuk dipakai pada langkah berikutnya.
            summary[f"sesuai_{format_distance(distance)}km"] = count
            # Menyimpan nilai ke `summary[f"hit_rate_{format_distance(distance)}km"]` untuk dipakai pada langkah berikutnya.
            summary[f"hit_rate_{format_distance(distance)}km"] = round(count / max(len(items), 1), 6)
        # Melanjutkan langkah kerja pada bagian kode ini.
        summary_rows.append(summary)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return summary_rows


# Membuat langkah kerja bernama `format_distance`.
def format_distance(distance: float) -> str:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return str(int(distance)) if float(distance).is_integer() else str(distance).replace(".", "_")


# Membuat langkah kerja bernama `draw_overlay`.
def draw_overlay(
    # Menjelaskan data `path` yang disimpan atau dikirim pada bagian ini.
    path: Path,
    # Menjelaskan data `base_image` yang disimpan atau dikirim pada bagian ini.
    base_image: Image.Image | None,
    # Menjelaskan data `probability` yang disimpan atau dikirim pada bagian ini.
    probability: np.ndarray | None,
    # Menjelaskan data `detail_rows` yang disimpan atau dikirim pada bagian ini.
    detail_rows: list[dict[str, Any]],
    # Menjelaskan data `admin_features` yang disimpan atau dikirim pada bagian ini.
    admin_features: list[dict[str, Any]],
    # Menjelaskan data `args` yang disimpan atau dikirim pada bagian ini.
    args: argparse.Namespace,
    # Menjelaskan data `distance_for_color` yang disimpan atau dikirim pada bagian ini.
    distance_for_color: float,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> None:
    # Menyimpan nilai ke `width` untuk dipakai pada langkah berikutnya.
    width = args.image_width
    # Menyimpan nilai ke `height` untuk dipakai pada langkah berikutnya.
    height = args.image_height
    # Mengecek syarat sebelum melanjutkan proses.
    if base_image is None:
        # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
        image = Image.new("RGBA", (width, height), (245, 241, 232, 255))
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
        image = base_image.convert("RGBA").resize((width, height))

    # Mengecek syarat sebelum melanjutkan proses.
    if probability is not None:
        # Menggabungkan hasil deteksi merah menjadi mask hotspot.
        mask = probability >= args.threshold
        # Menyimpan nilai ke `overlay` untuk dipakai pada langkah berikutnya.
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # Menyimpan nilai ke `alpha` untuk dipakai pada langkah berikutnya.
        alpha = np.zeros((height, width), dtype=np.uint8)
        # Menyimpan nilai ke `alpha[mask]` untuk dipakai pada langkah berikutnya.
        alpha[mask] = 90
        # Menyimpan nilai ke `color` untuk dipakai pada langkah berikutnya.
        color = np.zeros((height, width, 4), dtype=np.uint8)
        # Menyimpan nilai ke `color[..., 0]` untuk dipakai pada langkah berikutnya.
        color[..., 0] = 255
        # Menyimpan nilai ke `color[..., 1]` untuk dipakai pada langkah berikutnya.
        color[..., 1] = 165
        # Menyimpan nilai ke `color[..., 2]` untuk dipakai pada langkah berikutnya.
        color[..., 2] = 0
        # Menyimpan nilai ke `color[..., 3]` untuk dipakai pada langkah berikutnya.
        color[..., 3] = alpha
        # Menyimpan nilai ke `overlay` untuk dipakai pada langkah berikutnya.
        overlay = Image.fromarray(color)
        # Melanjutkan langkah kerja pada bagian kode ini.
        image.alpha_composite(overlay)

    # Menyimpan nilai ke `draw` untuk dipakai pada langkah berikutnya.
    draw = ImageDraw.Draw(image)
    # Mengulang proses untuk setiap data dalam daftar.
    for feature in admin_features:
        # Mengulang proses untuk setiap data dalam daftar.
        for path_pixels in geometry_pixel_paths(
            # Melanjutkan langkah kerja pada bagian kode ini.
            feature["geometry"],
            # Menyimpan nilai ke `center_lon` untuk dipakai pada langkah berikutnya.
            center_lon=args.center_lon,
            # Menyimpan nilai ke `center_lat` untuk dipakai pada langkah berikutnya.
            center_lat=args.center_lat,
            # Menyimpan nilai ke `zoom` untuk dipakai pada langkah berikutnya.
            zoom=args.zoom,
            # Menyimpan nilai ke `image_width` untuk dipakai pada langkah berikutnya.
            image_width=width,
            # Menyimpan nilai ke `image_height` untuk dipakai pada langkah berikutnya.
            image_height=height,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ):
            # Mengecek syarat sebelum melanjutkan proses.
            if len(path_pixels) >= 2:
                # Menyimpan nilai ke `draw.line(path_pixels, fill` untuk dipakai pada langkah berikutnya.
                draw.line(path_pixels, fill=(25, 25, 25, 180), width=1)

    # Menyimpan nilai ke `match_field` untuk dipakai pada langkah berikutnya.
    match_field = f"match_{format_distance(distance_for_color)}km"
    # Mengulang proses untuk setiap data dalam daftar.
    for row in detail_rows:
        # Menyimpan nilai ke `x` untuk dipakai pada langkah berikutnya.
        x = float(row["pixel_x"])
        # Menyimpan nilai ke `y` untuk dipakai pada langkah berikutnya.
        y = float(row["pixel_y"])
        # Mengecek syarat sebelum melanjutkan proses.
        if not (0 <= x < width and 0 <= y < height):
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Mengecek syarat sebelum melanjutkan proses.
        if probability is None:
            # Menyimpan nilai ke `color` untuk dipakai pada langkah berikutnya.
            color = (0, 90, 255, 255)
        # Mengecek syarat lain jika syarat sebelumnya tidak terpenuhi.
        elif str(row.get(match_field, "")).lower() == "true":
            # Menyimpan nilai ke `color` untuk dipakai pada langkah berikutnya.
            color = (0, 180, 80, 255)
        # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
        else:
            # Menyimpan nilai ke `color` untuk dipakai pada langkah berikutnya.
            color = (220, 40, 40, 255)
        # Menyimpan nilai ke `radius` untuk dipakai pada langkah berikutnya.
        radius = 4
        # Menyimpan nilai ke `draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill` untuk dipakai pada langkah berikutnya.
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0, 255))

    # Menyimpan nilai ke `path.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Melanjutkan langkah kerja pada bagian kode ini.
    image.save(path)


# Membuat langkah kerja bernama `build_markdown_report`.
def build_markdown_report(
    # Menjelaskan data `path` yang disimpan atau dikirim pada bagian ini.
    path: Path,
    # Menjelaskan data `args` yang disimpan atau dikirim pada bagian ini.
    args: argparse.Namespace,
    # Menjelaskan data `summary` yang disimpan atau dikirim pada bagian ini.
    summary: dict[str, Any],
    # Menjelaskan data `output_files` yang disimpan atau dikirim pada bagian ini.
    output_files: dict[str, str],
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> None:
    # Menyimpan nilai ke `lines` untuk dipakai pada langkah berikutnya.
    lines = [
        # Melanjutkan langkah kerja pada bagian kode ini.
        "# Laporan Evaluasi Geospasial FIRMS",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Dokumen ini dihasilkan otomatis sebagai bukti pendamping evaluasi model.",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Evaluasi ini tidak mengganti evaluasi raster/piksel yang sudah ada, tetapi",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "menambahkan pembuktian berbasis titik koordinat FIRMS.",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "## Konfigurasi",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- CSV FIRMS: `{args.firms_csv}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Tanggal target: `{args.target_date or 'semua tanggal pada CSV'}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Probability NPY: `{args.probability_npy or 'tidak digunakan'}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Threshold: `{args.threshold}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Center peta: `{args.center_lon}, {args.center_lat}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Zoom: `{args.zoom}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Ukuran gambar: `{args.image_width} x {args.image_height}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Km per piksel estimasi: `{summary['km_per_pixel']}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Batas administrasi: `{args.admin_geojson or 'tidak digunakan'}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "## Ringkasan Hasil",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Jumlah titik FIRMS dibaca: `{summary['total_firms_rows']}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Titik di dalam cakupan gambar: `{summary['points_inside_image']}`",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"- Titik di luar cakupan gambar: `{summary['points_outside_image']}`",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ]
    # Mengecek syarat sebelum melanjutkan proses.
    if summary.get("prediction_used"):
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        lines.extend(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            [
                # Melanjutkan langkah kerja pada bagian kode ini.
                f"- Piksel area prediksi di atas threshold: `{summary['prediction_positive_pixels']}`",
                # Melanjutkan langkah kerja pada bagian kode ini.
                f"- Persentase area prediksi: `{summary['prediction_positive_ratio']}`",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            ]
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Mengulang proses untuk setiap data dalam daftar.
        for distance, stats in summary["distance_summary"].items():
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            lines.append(
                # Melanjutkan langkah kerja pada bagian kode ini.
                f"- Radius {distance} km: `{stats['matched_points']}` titik sesuai "
                # Menyimpan nilai ke `f"(`hit_rate` untuk dipakai pada langkah berikutnya.
                f"(`hit_rate={stats['hit_rate']}`)"
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Melanjutkan langkah kerja pada bagian kode ini.
        lines.append("- Evaluasi jarak prediksi belum dihitung karena `--probability-npy` tidak diberikan.")

    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    lines.extend(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            # Melanjutkan langkah kerja pada bagian kode ini.
            "",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "## File Output",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengulang proses untuk setiap data dalam daftar.
    for label, file_path in output_files.items():
        # Melanjutkan langkah kerja pada bagian kode ini.
        lines.append(f"- {label}: `{file_path}`")

    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    lines.extend(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            # Melanjutkan langkah kerja pada bagian kode ini.
            "",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "## Batas Interpretasi",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "- Titik FIRMS merupakan pusat piksel deteksi satelit, bukan posisi api lapangan yang presisi.",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "- Validasi radius 1 km, 3 km, dan 5 km adalah validasi toleransi spasial, bukan validasi ground check lapangan.",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "- Nama kabupaten/kecamatan hanya kuat jika diperoleh dari file batas administrasi geospasial.",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "- Jika batas administrasi tidak diberikan, tabel tetap memuat koordinat dan atribut FIRMS, tetapi belum memuat nama wilayah administratif.",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `path.write_text("\n".join(lines) + "\n", encoding` untuk dipakai pada langkah berikutnya.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Membuat langkah kerja bernama `main`.
def main() -> None:
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parse_args()
    # Menyimpan nilai ke `distances` untuk dipakai pada langkah berikutnya.
    distances = split_distances(args.distance_km)
    # Menyimpan nilai ke `max_distance` untuk dipakai pada langkah berikutnya.
    max_distance = max(distances)
    # Menyimpan nilai ke `km_px` untuk dipakai pada langkah berikutnya.
    km_px = km_per_pixel(args.center_lat, args.zoom)
    # Menyimpan nilai ke `max_radius_px` untuk dipakai pada langkah berikutnya.
    max_radius_px = max(1, int(math.ceil(max_distance / km_px)))

    # Menyimpan nilai ke `firms_fieldnames, firms_rows` untuk dipakai pada langkah berikutnya.
    firms_fieldnames, firms_rows = read_firms_rows(args.firms_csv, args.target_date)
    # Menyimpan nilai ke `admin_features` untuk dipakai pada langkah berikutnya.
    admin_features = load_admin_features(
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.admin_geojson,
        # Melanjutkan langkah kerja pada bagian kode ini.
        split_fields(args.kabupaten_fields),
        # Melanjutkan langkah kerja pada bagian kode ini.
        split_fields(args.kecamatan_fields),
        # Melanjutkan langkah kerja pada bagian kode ini.
        args.max_boundary_features,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
    probability = load_probability(args.probability_npy)
    # Mengecek syarat sebelum melanjutkan proses.
    if probability is not None:
        # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
        probability = resize_probability(probability, args.image_width, args.image_height)
        # Menyimpan nilai ke `prediction_mask` untuk dipakai pada langkah berikutnya.
        prediction_mask = probability >= args.threshold
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menyimpan nilai ke `prediction_mask` untuk dipakai pada langkah berikutnya.
        prediction_mask = None

    # Menyimpan nilai ke `target_image_path` untuk dipakai pada langkah berikutnya.
    target_image_path = args.target_image
    # Mengecek syarat sebelum melanjutkan proses.
    if target_image_path is None and args.target_date:
        # Menyimpan nilai ke `target_image_path` untuk dipakai pada langkah berikutnya.
        target_image_path = find_target_image(args.dataset_dir, args.target_date)
    # Menyimpan nilai ke `base_image` untuk dipakai pada langkah berikutnya.
    base_image = Image.open(target_image_path) if target_image_path and target_image_path.exists() else None

    # Menyimpan nilai ke `detail_rows` untuk dipakai pada langkah berikutnya.
    detail_rows: list[dict[str, Any]] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for row in firms_rows:
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Menyimpan nilai ke `lat` untuk dipakai pada langkah berikutnya.
            lat = float(row["latitude"])
            # Menyimpan nilai ke `lon` untuk dipakai pada langkah berikutnya.
            lon = float(row["longitude"])
        # Menangani kesalahan agar program tidak langsung berhenti.
        except (KeyError, TypeError, ValueError):
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Menyimpan nilai ke `pixel_x, pixel_y` untuk dipakai pada langkah berikutnya.
        pixel_x, pixel_y = lonlat_to_pixel(
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            lon,
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            lat,
            # Menyimpan nilai ke `center_lon` untuk dipakai pada langkah berikutnya.
            center_lon=args.center_lon,
            # Menyimpan nilai ke `center_lat` untuk dipakai pada langkah berikutnya.
            center_lat=args.center_lat,
            # Menyimpan nilai ke `zoom` untuk dipakai pada langkah berikutnya.
            zoom=args.zoom,
            # Menyimpan nilai ke `image_width` untuk dipakai pada langkah berikutnya.
            image_width=args.image_width,
            # Menyimpan nilai ke `image_height` untuk dipakai pada langkah berikutnya.
            image_height=args.image_height,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `inside_image` untuk dipakai pada langkah berikutnya.
        inside_image = 0 <= pixel_x < args.image_width and 0 <= pixel_y < args.image_height
        # Menyimpan nilai ke `kabupaten, kecamatan` untuk dipakai pada langkah berikutnya.
        kabupaten, kecamatan = spatial_join(lon, lat, admin_features) if admin_features else ("", "")

        # Menyimpan nilai ke `sampled_probability` untuk dipakai pada langkah berikutnya.
        sampled_probability = ""
        # Menyimpan nilai ke `min_distance_px` untuk dipakai pada langkah berikutnya.
        min_distance_px = ""
        # Menyimpan nilai ke `min_distance_km` untuk dipakai pada langkah berikutnya.
        min_distance_km = ""
        # Menyimpan nilai ke `matches` untuk dipakai pada langkah berikutnya.
        matches: dict[str, bool | str] = {
            # Melanjutkan langkah kerja pada bagian kode ini.
            f"match_{format_distance(distance)}km": "" for distance in distances
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        }
        # Mengecek syarat sebelum melanjutkan proses.
        if prediction_mask is not None and inside_image:
            # Menyimpan nilai ke `xi` untuk dipakai pada langkah berikutnya.
            xi = int(round(pixel_x))
            # Menyimpan nilai ke `yi` untuk dipakai pada langkah berikutnya.
            yi = int(round(pixel_y))
            # Menyimpan nilai ke `xi` untuk dipakai pada langkah berikutnya.
            xi = min(max(xi, 0), args.image_width - 1)
            # Menyimpan nilai ke `yi` untuk dipakai pada langkah berikutnya.
            yi = min(max(yi, 0), args.image_height - 1)
            # Menyimpan nilai ke `sampled_probability` untuk dipakai pada langkah berikutnya.
            sampled_probability = round(float(probability[yi, xi]), 6)
            # Menyimpan nilai ke `distance_px` untuk dipakai pada langkah berikutnya.
            distance_px = nearest_positive_distance_px(prediction_mask, pixel_x, pixel_y, max_radius_px)
            # Mengecek syarat sebelum melanjutkan proses.
            if distance_px is not None:
                # Menyimpan nilai ke `min_distance_px` untuk dipakai pada langkah berikutnya.
                min_distance_px = round(distance_px, 4)
                # Menyimpan nilai ke `min_distance_km_value` untuk dipakai pada langkah berikutnya.
                min_distance_km_value = distance_px * km_px
                # Menyimpan nilai ke `min_distance_km` untuk dipakai pada langkah berikutnya.
                min_distance_km = round(min_distance_km_value, 4)
                # Menyimpan nilai ke `matches` untuk dipakai pada langkah berikutnya.
                matches = {
                    # Menyimpan nilai ke `f"match_{format_distance(distance)}km"` untuk dipakai pada langkah berikutnya.
                    f"match_{format_distance(distance)}km": min_distance_km_value <= distance
                    # Mengulang proses untuk setiap data dalam daftar.
                    for distance in distances
                # Menutup susunan data atau perintah yang dimulai sebelumnya.
                }
            # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
            else:
                # Menyimpan nilai ke `matches` untuk dipakai pada langkah berikutnya.
                matches = {f"match_{format_distance(distance)}km": False for distance in distances}

        # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
        detail = dict(row)
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        detail.update(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                # Melanjutkan langkah kerja pada bagian kode ini.
                "longitude": row.get("longitude", ""),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "latitude": row.get("latitude", ""),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "pixel_x": round(pixel_x, 3),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "pixel_y": round(pixel_y, 3),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "inside_image": inside_image,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "kabupaten": kabupaten,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "kecamatan": kecamatan,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "sampled_probability": sampled_probability,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "min_distance_px": min_distance_px,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "min_distance_km": min_distance_km,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            }
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Melanjutkan langkah kerja pada bagian kode ini.
        detail.update(matches)
        # Melanjutkan langkah kerja pada bagian kode ini.
        detail_rows.append(detail)

    # Menyimpan nilai ke `args.output_dir.mkdir(parents` untuk dipakai pada langkah berikutnya.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `date_suffix` untuk dipakai pada langkah berikutnya.
    date_suffix = args.target_date or "all_dates"
    # Menyimpan nilai ke `detail_path` untuk dipakai pada langkah berikutnya.
    detail_path = args.output_dir / f"validasi_geospasial_detail_{date_suffix}.csv"
    # Menyimpan nilai ke `summary_path` untuk dipakai pada langkah berikutnya.
    summary_path = args.output_dir / f"validasi_geospasial_summary_{date_suffix}.json"
    # Menyimpan nilai ke `kabupaten_path` untuk dipakai pada langkah berikutnya.
    kabupaten_path = args.output_dir / f"rekap_per_kabupaten_{date_suffix}.csv"
    # Menyimpan nilai ke `kecamatan_path` untuk dipakai pada langkah berikutnya.
    kecamatan_path = args.output_dir / f"rekap_per_kecamatan_{date_suffix}.csv"
    # Menyimpan nilai ke `overlay_path` untuk dipakai pada langkah berikutnya.
    overlay_path = args.output_dir / f"overlay_prediksi_vs_firms_{date_suffix}.png"
    # Menyimpan nilai ke `report_path` untuk dipakai pada langkah berikutnya.
    report_path = args.output_dir / f"LAPORAN_VALIDASI_GEOSPASIAL_FIRMS_{date_suffix}.md"

    # Menyimpan nilai ke `extra_fields` untuk dipakai pada langkah berikutnya.
    extra_fields = [
        # Melanjutkan langkah kerja pada bagian kode ini.
        "pixel_x",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "pixel_y",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "inside_image",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "kabupaten",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "kecamatan",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "sampled_probability",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "min_distance_px",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "min_distance_km",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ] + [f"match_{format_distance(distance)}km" for distance in distances]
    # Menyimpan nilai ke `detail_fields` untuk dipakai pada langkah berikutnya.
    detail_fields = list(dict.fromkeys(firms_fieldnames + extra_fields))
    # Melanjutkan langkah kerja pada bagian kode ini.
    write_csv_safe(detail_path, detail_fields, detail_rows)

    # Menyimpan nilai ke `inside_count` untuk dipakai pada langkah berikutnya.
    inside_count = sum(1 for row in detail_rows if row["inside_image"])
    # Menyimpan nilai ke `outside_count` untuk dipakai pada langkah berikutnya.
    outside_count = len(detail_rows) - inside_count
    # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
    summary: dict[str, Any] = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "firms_csv": str(args.firms_csv),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "target_date": args.target_date or None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "target_image": str(target_image_path) if target_image_path else None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "admin_geojson": str(args.admin_geojson) if args.admin_geojson else None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "probability_npy": str(args.probability_npy) if args.probability_npy else None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "threshold": args.threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "km_per_pixel": round(km_px, 6),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "max_search_radius_px": max_radius_px,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "total_firms_rows": len(firms_rows),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "detail_rows_written": len(detail_rows),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "points_inside_image": inside_count,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "points_outside_image": outside_count,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "admin_features_loaded": len(admin_features),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "prediction_used": probability is not None,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Mengecek syarat sebelum melanjutkan proses.
    if probability is not None and prediction_mask is not None:
        # Menyimpan nilai ke `positive_pixels` untuk dipakai pada langkah berikutnya.
        positive_pixels = int(np.count_nonzero(prediction_mask))
        # Menyimpan nilai ke `summary["prediction_positive_pixels"]` untuk dipakai pada langkah berikutnya.
        summary["prediction_positive_pixels"] = positive_pixels
        # Menyimpan nilai ke `summary["prediction_positive_ratio"]` untuk dipakai pada langkah berikutnya.
        summary["prediction_positive_ratio"] = round(positive_pixels / prediction_mask.size, 8)
        # Menyimpan nilai ke `distance_summary` untuk dipakai pada langkah berikutnya.
        distance_summary: dict[str, dict[str, Any]] = {}
        # Mengulang proses untuk setiap data dalam daftar.
        for distance in distances:
            # Menyimpan nilai ke `field` untuk dipakai pada langkah berikutnya.
            field = f"match_{format_distance(distance)}km"
            # Menyimpan nilai ke `matched` untuk dipakai pada langkah berikutnya.
            matched = sum(1 for row in detail_rows if row.get(field) is True)
            # Menyimpan nilai ke `distance_summary[format_distance(distance)]` untuk dipakai pada langkah berikutnya.
            distance_summary[format_distance(distance)] = {
                # Melanjutkan langkah kerja pada bagian kode ini.
                "radius_km": distance,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "matched_points": matched,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "evaluated_points": inside_count,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "hit_rate": round(matched / max(inside_count, 1), 6),
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            }
        # Menyimpan nilai ke `summary["distance_summary"]` untuk dipakai pada langkah berikutnya.
        summary["distance_summary"] = distance_summary

    # Menyimpan nilai ke `summary_path.write_text(json.dumps(summary, indent` untuk dipakai pada langkah berikutnya.
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Mengecek syarat sebelum melanjutkan proses.
    if any(row.get("kabupaten") for row in detail_rows):
        # Menyimpan nilai ke `kabupaten_rows` untuk dipakai pada langkah berikutnya.
        kabupaten_rows = summarize_by(detail_rows, "kabupaten", distances)
        # Melanjutkan langkah kerja pada bagian kode ini.
        write_csv_safe(kabupaten_path, list(kabupaten_rows[0].keys()) if kabupaten_rows else ["kabupaten"], kabupaten_rows)
    # Mengecek syarat sebelum melanjutkan proses.
    if any(row.get("kecamatan") for row in detail_rows):
        # Menyimpan nilai ke `kecamatan_rows` untuk dipakai pada langkah berikutnya.
        kecamatan_rows = summarize_by(detail_rows, "kecamatan", distances)
        # Melanjutkan langkah kerja pada bagian kode ini.
        write_csv_safe(kecamatan_path, list(kecamatan_rows[0].keys()) if kecamatan_rows else ["kecamatan"], kecamatan_rows)

    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    draw_overlay(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        overlay_path,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        base_image,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        probability,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        detail_rows,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        admin_features,
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        args,
        # Menyimpan nilai ke `distance_for_color` untuk dipakai pada langkah berikutnya.
        distance_for_color=distances[min(1, len(distances) - 1)],
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `output_files` untuk dipakai pada langkah berikutnya.
    output_files = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Detail titik": str(detail_path),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Ringkasan JSON": str(summary_path),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Overlay": str(overlay_path),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Mengecek syarat sebelum melanjutkan proses.
    if kabupaten_path.exists():
        # Menyimpan nilai ke `output_files["Rekap kabupaten"]` untuk dipakai pada langkah berikutnya.
        output_files["Rekap kabupaten"] = str(kabupaten_path)
    # Mengecek syarat sebelum melanjutkan proses.
    if kecamatan_path.exists():
        # Menyimpan nilai ke `output_files["Rekap kecamatan"]` untuk dipakai pada langkah berikutnya.
        output_files["Rekap kecamatan"] = str(kecamatan_path)

    # Melanjutkan langkah kerja pada bagian kode ini.
    build_markdown_report(report_path, args, summary, output_files)
    # Menyimpan nilai ke `output_files["Laporan Markdown"]` untuk dipakai pada langkah berikutnya.
    output_files["Laporan Markdown"] = str(report_path)

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[geospatial] Selesai")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[geospatial] Titik FIRMS: {len(firms_rows)}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[geospatial] Titik dalam gambar: {inside_count}")
    # Mengecek syarat sebelum melanjutkan proses.
    if probability is not None:
        # Mengulang proses untuk setiap data dalam daftar.
        for distance, stats in summary["distance_summary"].items():
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print(
                # Melanjutkan langkah kerja pada bagian kode ini.
                f"[geospatial] Radius {distance} km: "
                # Melanjutkan langkah kerja pada bagian kode ini.
                f"{stats['matched_points']}/{stats['evaluated_points']} "
                # Menyimpan nilai ke `f"(hit_rate` untuk dipakai pada langkah berikutnya.
                f"(hit_rate={stats['hit_rate']})"
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
    # Mengulang proses untuk setiap data dalam daftar.
    for label, file_path in output_files.items():
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"[geospatial] {label}: {Path(file_path).resolve()}")


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Melanjutkan langkah kerja pada bagian kode ini.
    main()
