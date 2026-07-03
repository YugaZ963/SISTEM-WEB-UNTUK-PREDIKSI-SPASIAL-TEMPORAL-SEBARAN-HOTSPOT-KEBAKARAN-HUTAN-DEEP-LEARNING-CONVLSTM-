
"""Komentar file skripsi:
Tool evaluasi geospasial untuk membandingkan titik FIRMS aktual dengan area prediksi model.

This tool is intentionally separate from training and the web backend. It adds
the geospatial evidence requested for the thesis: FIRMS point attributes,
optional administrative spatial join, distance-based validation, and overlay
figures.

It avoids heavy GIS dependencies. Administrative boundaries are accepted as
GeoJSON so point-in-polygon can be done with a small local implementation.

Konteks laporan: file ini mendukung tahapan Evaluation pada BAB IV,
khususnya pembuktian data atribut, data vektor, dan validasi jarak spasial.
"""

from __future__ import annotations

# argparse dipakai agar parameter dataset, patch, threshold, dan output bisa diatur dari command line.
import argparse
# csv dipakai untuk membaca titik FIRMS mentah saat validasi geospasial.
import csv
# json dipakai untuk menyimpan ringkasan eksperimen, metrik, dan metadata proses.
import json
# math dipakai untuk perhitungan ukuran patch, skor, atau pembulatan metrik numerik.
import math
from collections import defaultdict
# datetime dipakai untuk menjaga urutan kronologis citra dan menghitung target H+1.
from datetime import datetime, timezone
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path
from typing import Any

# NumPy dipakai untuk mask hotspot, tensor sequence, patch, probability map, dan metrik piksel.
import numpy as np
# Pillow Image dipakai untuk membuka citra hotspot PNG/JPG dan menyimpan preview/hasil.
from PIL import Image, ImageDraw


DEFAULT_FIRMS_CSV = Path(
    "data-firms-mentah-pendamping/"
    "modis_sp_2025-06-21_2025-11-25_cakupan_peta_project.csv"
)
DEFAULT_DATASET_DIR = Path("Ipynb/Dataset History Fire Hotspot In Riau Province PNG")
DEFAULT_OUTPUT_DIR = Path("artifacts/geospatial_firms_validation")

WEB_MERCATOR_MAX_LAT = 85.05112878
EARTH_EQUATORIAL_CIRCUMFERENCE_M = 40075016.68557849
TILE_SIZE = 256.0


# Membaca semua parameter training seperti dataset, patch, threshold, batch, epoch, dan lokasi output.
def parse_args() -> argparse.Namespace:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description=(
            "Evaluasi geospasial titik FIRMS terhadap peta prediksi risiko "
            "hotspot H+1."
        )
    )
    # Opsi `--firms-csv` menambah parameter eksekusi script.
    parser.add_argument("--firms-csv", type=Path, default=DEFAULT_FIRMS_CSV)
    parser.add_argument(
        "--target-date",
        default="",
        help="Filter tanggal aktual FIRMS, contoh 2025-07-19. Kosongkan untuk semua tanggal.",
    )
    parser.add_argument(
        "--probability-npy",
        type=Path,
        default=None,
        help="File probability_H+1.npy hasil model. Jika kosong, hanya dibuat tabel titik/overlay FIRMS.",
    )
    parser.add_argument(
        "--target-image",
        type=Path,
        default=None,
        help="PNG/JPG tanggal target untuk background overlay. Jika kosong, dicari dari dataset-dir.",
    )
    # Opsi ini menentukan folder citra hotspot historis yang akan diproses.
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--admin-geojson",
        type=Path,
        default=None,
        help="GeoJSON batas administrasi kabupaten/kecamatan untuk spatial join.",
    )
    # Opsi ini menentukan ambang binary mask pada evaluasi atau visualisasi.
    parser.add_argument("--threshold", type=float, default=0.55)
    # Opsi `--center-lon` menambah parameter eksekusi script.
    parser.add_argument("--center-lon", type=float, default=102.1)
    # Opsi `--center-lat` menambah parameter eksekusi script.
    parser.add_argument("--center-lat", type=float, default=0.8)
    # Opsi `--zoom` menambah parameter eksekusi script.
    parser.add_argument("--zoom", type=float, default=8.1)
    # Opsi `--image-width` menambah parameter eksekusi script.
    parser.add_argument("--image-width", type=int, default=1528)
    # Opsi `--image-height` menambah parameter eksekusi script.
    parser.add_argument("--image-height", type=int, default=773)
    parser.add_argument(
        "--distance-km",
        default="1,3,5",
        help="Daftar radius evaluasi dalam kilometer, contoh: 1,3,5.",
    )
    parser.add_argument(
        "--kabupaten-fields",
        default="kabupaten_kota,shapeName,NAME_2,WADMKK,KABUPATEN,KAB_KOTA,KABUPATEN_KOTA,KABKOT,NAMOBJ",
        help="Prioritas nama field kabupaten/kota pada GeoJSON.",
    )
    parser.add_argument(
        "--kecamatan-fields",
        default="NAME_3,WADMKC,KECAMATAN,KEC,NAMOBJ",
        help="Prioritas nama field kecamatan pada GeoJSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-boundary-features",
        type=int,
        default=0,
        help="Batas jumlah feature admin yang dibaca. 0 berarti semua.",
    )
    # Hasil ini dikembalikan sebagai output fungsi `parse_args` untuk tahap berikutnya.
    return parser.parse_args()


# Fungsi `split_fields` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def split_fields(value: str) -> list[str]:
    # Hasil ini dikembalikan sebagai output fungsi `split_fields` untuk tahap berikutnya.
    return [part.strip() for part in value.split(",") if part.strip()]


# Fungsi `split_distances` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def split_distances(value: str) -> list[float]:
    distances = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not distances:
        raise SystemExit("--distance-km minimal berisi satu nilai.")
    # Hasil ini dikembalikan sebagai output fungsi `split_distances` untuk tahap berikutnya.
    return sorted(distances)


# Fungsi `web_mercator_global_pixel` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def web_mercator_global_pixel(lon: float, lat: float, zoom: float) -> tuple[float, float]:
    lat = max(min(lat, WEB_MERCATOR_MAX_LAT), -WEB_MERCATOR_MAX_LAT)
    scale = TILE_SIZE * (2.0**zoom)
    x = (lon + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    # Hasil ini dikembalikan sebagai output fungsi `web_mercator_global_pixel` untuk tahap berikutnya.
    return x, y


# Fungsi `lonlat_to_pixel` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def lonlat_to_pixel(
    lon: float,
    lat: float,
    *,
    center_lon: float,
    center_lat: float,
    zoom: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    center_x, center_y = web_mercator_global_pixel(center_lon, center_lat, zoom)
    x, y = web_mercator_global_pixel(lon, lat, zoom)
    # Hasil ini dikembalikan sebagai output fungsi `lonlat_to_pixel` untuk tahap berikutnya.
    return x - center_x + image_width / 2.0, y - center_y + image_height / 2.0


# Fungsi `km_per_pixel` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def km_per_pixel(center_lat: float, zoom: float) -> float:
    meters = (
        EARTH_EQUATORIAL_CIRCUMFERENCE_M
        * math.cos(math.radians(center_lat))
        / (TILE_SIZE * (2.0**zoom))
    )
    # Hasil ini dikembalikan sebagai output fungsi `km_per_pixel` untuk tahap berikutnya.
    return meters / 1000.0


# Fungsi `read_firms_rows` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def read_firms_rows(path: Path, target_date: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV FIRMS tidak ditemukan: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"latitude", "longitude", "acq_date"}
        missing = required.difference(fieldnames)
        if missing:
            # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
            raise ValueError(f"CSV FIRMS tidak memiliki kolom wajib: {sorted(missing)}")
        rows = []
        for row in reader:
            if target_date and row.get("acq_date") != target_date:
                continue
            rows.append(row)
    # Hasil ini dikembalikan sebagai output fungsi `read_firms_rows` untuk tahap berikutnya.
    return fieldnames, rows


# Fungsi `load_probability` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def load_probability(path: Path | None) -> np.ndarray | None:
    if path is None:
        # Hasil ini dikembalikan sebagai output fungsi `load_probability` untuk tahap berikutnya.
        return None
    if not path.exists():
        raise FileNotFoundError(f"Probability NPY tidak ditemukan: {path}")
    # Peta probabilitas risiko hotspot dengan nilai 0 sampai 1.
    probability = np.load(path).astype(np.float32)
    # Peta probabilitas risiko hotspot dengan nilai 0 sampai 1.
    probability = np.squeeze(probability)
    if probability.ndim != 2:
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError(
            f"Probability NPY harus menjadi array 2D setelah squeeze, didapat {probability.shape}"
        )
    # Nilai dikunci ke rentang 0-1 agar cocok sebagai mask/probability map.
    return np.clip(probability, 0.0, 1.0)


# Fungsi `resize_probability` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def resize_probability(probability: np.ndarray, width: int, height: int) -> np.ndarray:
    if probability.shape == (height, width):
        # Hasil ini dikembalikan sebagai output fungsi `resize_probability` untuk tahap berikutnya.
        return probability
    # Nilai dikunci ke rentang 0-1 agar cocok sebagai mask/probability map.
    image = Image.fromarray((np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8))
    resized = image.resize((width, height), Image.BILINEAR)
    # Hasil ini dikembalikan sebagai output fungsi `resize_probability` untuk tahap berikutnya.
    return np.asarray(resized, dtype=np.float32) / 255.0


# Fungsi `find_target_image` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def find_target_image(dataset_dir: Path, target_date: str) -> Path | None:
    if not target_date or not dataset_dir.exists():
        # Hasil ini dikembalikan sebagai output fungsi `find_target_image` untuk tahap berikutnya.
        return None
    patterns = [
        f"FIRMS_{target_date}*.png",
        f"FIRMS_{target_date}*.jpg",
        f"FIRMS_{target_date}*.jpeg",
    ]
    for pattern in patterns:
        matches = sorted(dataset_dir.glob(pattern))
        if matches:
            # Hasil ini dikembalikan sebagai output fungsi `find_target_image` untuk tahap berikutnya.
            return matches[0]
    # Hasil ini dikembalikan sebagai output fungsi `find_target_image` untuk tahap berikutnya.
    return None


# Fungsi `point_in_ring` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        # Hasil ini dikembalikan sebagai output fungsi `point_in_ring` untuk tahap berikutnya.
        return False
    x, y = lon, lat
    previous_x, previous_y = ring[-1][0], ring[-1][1]
    for current in ring:
        current_x, current_y = current[0], current[1]
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            denom = previous_y - current_y
            if abs(denom) < 1e-12:
                intersection_x = current_x
            # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
            else:
                intersection_x = (
                    (previous_x - current_x) * (y - current_y) / denom + current_x
                )
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    # Hasil ini dikembalikan sebagai output fungsi `point_in_ring` untuk tahap berikutnya.
    return inside


# Fungsi `point_in_polygon` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def point_in_polygon(lon: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon:
        # Hasil ini dikembalikan sebagai output fungsi `point_in_polygon` untuk tahap berikutnya.
        return False
    if not point_in_ring(lon, lat, polygon[0]):
        # Hasil ini dikembalikan sebagai output fungsi `point_in_polygon` untuk tahap berikutnya.
        return False
    # Holes exclude the point.
    for hole in polygon[1:]:
        if point_in_ring(lon, lat, hole):
            # Hasil ini dikembalikan sebagai output fungsi `point_in_polygon` untuk tahap berikutnya.
            return False
    # Hasil ini dikembalikan sebagai output fungsi `point_in_polygon` untuk tahap berikutnya.
    return True


# Fungsi `geometry_contains_point` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def geometry_contains_point(geometry: dict[str, Any], lon: float, lat: float) -> bool:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        # Hasil ini dikembalikan sebagai output fungsi `geometry_contains_point` untuk tahap berikutnya.
        return point_in_polygon(lon, lat, coordinates)
    if geom_type == "MultiPolygon":
        # Hasil ini dikembalikan sebagai output fungsi `geometry_contains_point` untuk tahap berikutnya.
        return any(point_in_polygon(lon, lat, polygon) for polygon in coordinates)
    # Hasil ini dikembalikan sebagai output fungsi `geometry_contains_point` untuk tahap berikutnya.
    return False


# Fungsi `geometry_pixel_paths` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def geometry_pixel_paths(
    geometry: dict[str, Any],
    *,
    center_lon: float,
    center_lat: float,
    zoom: float,
    image_width: int,
    image_height: int,
) -> list[list[tuple[float, float]]]:
    paths: list[list[tuple[float, float]]] = []
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    polygons = [coordinates] if geom_type == "Polygon" else coordinates if geom_type == "MultiPolygon" else []
    for polygon in polygons:
        if not polygon:
            continue
        outer_ring = polygon[0]
        path = [
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            lonlat_to_pixel(
                lon,
                lat,
                center_lon=center_lon,
                center_lat=center_lat,
                zoom=zoom,
                image_width=image_width,
                image_height=image_height,
            )
            for lon, lat, *_ in outer_ring
        ]
        paths.append(path)
    # Hasil ini dikembalikan sebagai output fungsi `geometry_pixel_paths` untuk tahap berikutnya.
    return paths


# Fungsi `pick_property` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def pick_property(properties: dict[str, Any], fields: list[str]) -> str:
    normalized = {str(key).upper(): value for key, value in properties.items()}
    for field in fields:
        value = normalized.get(field.upper())
        if value not in (None, ""):
            # Hasil ini dikembalikan sebagai output fungsi `pick_property` untuk tahap berikutnya.
            return str(value)
    # Hasil ini dikembalikan sebagai output fungsi `pick_property` untuk tahap berikutnya.
    return ""


# Fungsi `load_admin_features` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def load_admin_features(
    path: Path | None,
    kabupaten_fields: list[str],
    kecamatan_fields: list[str],
    max_features: int,
) -> list[dict[str, Any]]:
    if path is None:
        # Hasil ini dikembalikan sebagai output fungsi `load_admin_features` untuk tahap berikutnya.
        return []
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON administrasi tidak ditemukan: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        # Error eksplisit ini membuat penyebab kegagalan data/parameter mudah dilacak.
        raise ValueError("GeoJSON administrasi harus bertipe FeatureCollection.")
    features = []
    for feature in data.get("features", []):
        properties = feature.get("properties", {}) or {}
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        features.append(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                "geometry": geometry,
                "properties": properties,
                "kabupaten": pick_property(properties, kabupaten_fields),
                "kecamatan": pick_property(properties, kecamatan_fields),
            }
        )
        if max_features and len(features) >= max_features:
            break
    # Hasil ini dikembalikan sebagai output fungsi `load_admin_features` untuk tahap berikutnya.
    return features


# Fungsi `spatial_join` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def spatial_join(lon: float, lat: float, features: list[dict[str, Any]]) -> tuple[str, str]:
    for feature in features:
        if geometry_contains_point(feature["geometry"], lon, lat):
            # Hasil ini dikembalikan sebagai output fungsi `spatial_join` untuk tahap berikutnya.
            return feature.get("kabupaten", ""), feature.get("kecamatan", "")
    # Hasil ini dikembalikan sebagai output fungsi `spatial_join` untuk tahap berikutnya.
    return "", ""


# Fungsi `nearest_positive_distance_px` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def nearest_positive_distance_px(
    mask: np.ndarray,
    x: float,
    y: float,
    max_radius_px: int,
) -> float | None:
    height, width = mask.shape
    center_x = int(round(x))
    center_y = int(round(y))
    left = max(0, center_x - max_radius_px)
    right = min(width - 1, center_x + max_radius_px)
    top = max(0, center_y - max_radius_px)
    bottom = min(height - 1, center_y + max_radius_px)
    if left > right or top > bottom:
        # Hasil ini dikembalikan sebagai output fungsi `nearest_positive_distance_px` untuk tahap berikutnya.
        return None
    window = mask[top : bottom + 1, left : right + 1]
    points = np.argwhere(window)
    if points.size == 0:
        # Hasil ini dikembalikan sebagai output fungsi `nearest_positive_distance_px` untuk tahap berikutnya.
        return None
    # np.argwhere returns y, x positions in the local window.
    local_y = points[:, 0] + top
    local_x = points[:, 1] + left
    distances = np.sqrt((local_x - x) ** 2 + (local_y - y) ** 2)
    # Hasil ini dikembalikan sebagai output fungsi `nearest_positive_distance_px` untuk tahap berikutnya.
    return float(np.min(distances))


# Fungsi `write_csv_safe` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def write_csv_safe(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Fungsi `summarize_by` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def summarize_by(rows: list[dict[str, Any]], group_field: str, distances: list[float]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_field) or "Tidak teridentifikasi")
        grouped[key].append(row)
    summary_rows = []
    for key, items in sorted(grouped.items()):
        # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
        summary: dict[str, Any] = {group_field: key, "jumlah_titik": len(items)}
        for distance in distances:
            field = f"match_{format_distance(distance)}km"
            count = sum(1 for item in items if str(item.get(field, "")).lower() == "true")
            summary[f"sesuai_{format_distance(distance)}km"] = count
            summary[f"hit_rate_{format_distance(distance)}km"] = round(count / max(len(items), 1), 6)
        summary_rows.append(summary)
    # Hasil ini dikembalikan sebagai output fungsi `summarize_by` untuk tahap berikutnya.
    return summary_rows


# Fungsi `format_distance` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def format_distance(distance: float) -> str:
    # Hasil ini dikembalikan sebagai output fungsi `format_distance` untuk tahap berikutnya.
    return str(int(distance)) if float(distance).is_integer() else str(distance).replace(".", "_")


# Fungsi `draw_overlay` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def draw_overlay(
    path: Path,
    base_image: Image.Image | None,
    probability: np.ndarray | None,
    detail_rows: list[dict[str, Any]],
    admin_features: list[dict[str, Any]],
    args: argparse.Namespace,
    distance_for_color: float,
) -> None:
    width = args.image_width
    height = args.image_height
    if base_image is None:
        image = Image.new("RGBA", (width, height), (245, 241, 232, 255))
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        image = base_image.convert("RGBA").resize((width, height))

    if probability is not None:
        # Mask biner hotspot: piksel merah menjadi positif dan background menjadi nol.
        mask = probability >= args.threshold
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        alpha = np.zeros((height, width), dtype=np.uint8)
        # `alpha[mask]` menyimpan representasi hotspot dalam bentuk mask numerik.
        alpha[mask] = 90
        color = np.zeros((height, width, 4), dtype=np.uint8)
        color[..., 0] = 255
        color[..., 1] = 165
        color[..., 2] = 0
        color[..., 3] = alpha
        overlay = Image.fromarray(color)
        image.alpha_composite(overlay)

    draw = ImageDraw.Draw(image)
    for feature in admin_features:
        for path_pixels in geometry_pixel_paths(
            feature["geometry"],
            center_lon=args.center_lon,
            center_lat=args.center_lat,
            zoom=args.zoom,
            image_width=width,
            image_height=height,
        ):
            if len(path_pixels) >= 2:
                draw.line(path_pixels, fill=(25, 25, 25, 180), width=1)

    match_field = f"match_{format_distance(distance_for_color)}km"
    for row in detail_rows:
        x = float(row["pixel_x"])
        y = float(row["pixel_y"])
        if not (0 <= x < width and 0 <= y < height):
            continue
        if probability is None:
            color = (0, 90, 255, 255)
        elif str(row.get(match_field, "")).lower() == "true":
            color = (0, 180, 80, 255)
        # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
        else:
            color = (220, 40, 40, 255)
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0, 255))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


# Fungsi `build_markdown_report` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def build_markdown_report(
    path: Path,
    args: argparse.Namespace,
    summary: dict[str, Any],
    output_files: dict[str, str],
) -> None:
    lines = [
        "# Laporan Evaluasi Geospasial FIRMS",
        "",
        "Dokumen ini dihasilkan otomatis sebagai bukti pendamping evaluasi model.",
        "Evaluasi ini tidak mengganti evaluasi raster/piksel yang sudah ada, tetapi",
        "menambahkan pembuktian berbasis titik koordinat FIRMS.",
        "",
        "## Konfigurasi",
        "",
        f"- CSV FIRMS: `{args.firms_csv}`",
        f"- Tanggal target: `{args.target_date or 'semua tanggal pada CSV'}`",
        f"- Probability NPY: `{args.probability_npy or 'tidak digunakan'}`",
        f"- Threshold: `{args.threshold}`",
        f"- Center peta: `{args.center_lon}, {args.center_lat}`",
        f"- Zoom: `{args.zoom}`",
        f"- Ukuran gambar: `{args.image_width} x {args.image_height}`",
        f"- Km per piksel estimasi: `{summary['km_per_pixel']}`",
        f"- Batas administrasi: `{args.admin_geojson or 'tidak digunakan'}`",
        "",
        "## Ringkasan Hasil",
        "",
        f"- Jumlah titik FIRMS dibaca: `{summary['total_firms_rows']}`",
        f"- Titik di dalam cakupan gambar: `{summary['points_inside_image']}`",
        f"- Titik di luar cakupan gambar: `{summary['points_outside_image']}`",
    ]
    if summary.get("prediction_used"):
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        lines.extend(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            [
                f"- Piksel area prediksi di atas threshold: `{summary['prediction_positive_pixels']}`",
                f"- Persentase area prediksi: `{summary['prediction_positive_ratio']}`",
            ]
        )
        for distance, stats in summary["distance_summary"].items():
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            lines.append(
                f"- Radius {distance} km: `{stats['matched_points']}` titik sesuai "
                f"(`hit_rate={stats['hit_rate']}`)"
            )
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        lines.append("- Evaluasi jarak prediksi belum dihitung karena `--probability-npy` tidak diberikan.")

    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    lines.extend(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            "",
            "## File Output",
            "",
        ]
    )
    for label, file_path in output_files.items():
        lines.append(f"- {label}: `{file_path}`")

    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    lines.extend(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            "",
            "## Batas Interpretasi",
            "",
            "- Titik FIRMS merupakan pusat piksel deteksi satelit, bukan posisi api lapangan yang presisi.",
            "- Validasi radius 1 km, 3 km, dan 5 km adalah validasi toleransi spasial, bukan validasi ground check lapangan.",
            "- Nama kabupaten/kecamatan hanya kuat jika diperoleh dari file batas administrasi geospasial.",
            "- Jika batas administrasi tidak diberikan, tabel tetap memuat koordinat dan atribut FIRMS, tetapi belum memuat nama wilayah administratif.",
        ]
    )
    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> None:
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parse_args()
    distances = split_distances(args.distance_km)
    max_distance = max(distances)
    km_px = km_per_pixel(args.center_lat, args.zoom)
    max_radius_px = max(1, int(math.ceil(max_distance / km_px)))

    firms_fieldnames, firms_rows = read_firms_rows(args.firms_csv, args.target_date)
    admin_features = load_admin_features(
        args.admin_geojson,
        split_fields(args.kabupaten_fields),
        split_fields(args.kecamatan_fields),
        args.max_boundary_features,
    )

    # Peta probabilitas risiko hotspot dengan nilai 0 sampai 1.
    probability = load_probability(args.probability_npy)
    if probability is not None:
        # Peta probabilitas risiko hotspot dengan nilai 0 sampai 1.
        probability = resize_probability(probability, args.image_width, args.image_height)
        # `prediction_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
        prediction_mask = probability >= args.threshold
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # `prediction_mask` menyimpan representasi hotspot dalam bentuk mask numerik.
        prediction_mask = None

    # `target_image_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    target_image_path = args.target_image
    if target_image_path is None and args.target_date:
        # `target_image_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
        target_image_path = find_target_image(args.dataset_dir, args.target_date)
    # Citra dibuka dengan Pillow untuk validasi, ekstraksi mask, atau pembuatan preview.
    base_image = Image.open(target_image_path) if target_image_path and target_image_path.exists() else None

    detail_rows: list[dict[str, Any]] = []
    for row in firms_rows:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        pixel_x, pixel_y = lonlat_to_pixel(
            lon,
            lat,
            center_lon=args.center_lon,
            center_lat=args.center_lat,
            zoom=args.zoom,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        inside_image = 0 <= pixel_x < args.image_width and 0 <= pixel_y < args.image_height
        kabupaten, kecamatan = spatial_join(lon, lat, admin_features) if admin_features else ("", "")

        sampled_probability = ""
        min_distance_px = ""
        min_distance_km = ""
        matches: dict[str, bool | str] = {
            f"match_{format_distance(distance)}km": "" for distance in distances
        }
        if prediction_mask is not None and inside_image:
            xi = int(round(pixel_x))
            yi = int(round(pixel_y))
            xi = min(max(xi, 0), args.image_width - 1)
            yi = min(max(yi, 0), args.image_height - 1)
            sampled_probability = round(float(probability[yi, xi]), 6)
            distance_px = nearest_positive_distance_px(prediction_mask, pixel_x, pixel_y, max_radius_px)
            if distance_px is not None:
                min_distance_px = round(distance_px, 4)
                min_distance_km_value = distance_px * km_px
                min_distance_km = round(min_distance_km_value, 4)
                matches = {
                    f"match_{format_distance(distance)}km": min_distance_km_value <= distance
                    for distance in distances
                }
            # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
            else:
                matches = {f"match_{format_distance(distance)}km": False for distance in distances}

        detail = dict(row)
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        detail.update(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                "longitude": row.get("longitude", ""),
                "latitude": row.get("latitude", ""),
                "pixel_x": round(pixel_x, 3),
                "pixel_y": round(pixel_y, 3),
                "inside_image": inside_image,
                "kabupaten": kabupaten,
                "kecamatan": kecamatan,
                "sampled_probability": sampled_probability,
                "min_distance_px": min_distance_px,
                "min_distance_km": min_distance_km,
            }
        )
        detail.update(matches)
        detail_rows.append(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    date_suffix = args.target_date or "all_dates"
    # `detail_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    detail_path = args.output_dir / f"validasi_geospasial_detail_{date_suffix}.csv"
    # `summary_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    summary_path = args.output_dir / f"validasi_geospasial_summary_{date_suffix}.json"
    # `kabupaten_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    kabupaten_path = args.output_dir / f"rekap_per_kabupaten_{date_suffix}.csv"
    # `kecamatan_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    kecamatan_path = args.output_dir / f"rekap_per_kecamatan_{date_suffix}.csv"
    # `overlay_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    overlay_path = args.output_dir / f"overlay_prediksi_vs_firms_{date_suffix}.png"
    # `report_path` menyimpan lokasi file output/input yang dipakai pada tahap ini.
    report_path = args.output_dir / f"LAPORAN_VALIDASI_GEOSPASIAL_FIRMS_{date_suffix}.md"

    extra_fields = [
        "pixel_x",
        "pixel_y",
        "inside_image",
        "kabupaten",
        "kecamatan",
        "sampled_probability",
        "min_distance_px",
        "min_distance_km",
    ] + [f"match_{format_distance(distance)}km" for distance in distances]
    detail_fields = list(dict.fromkeys(firms_fieldnames + extra_fields))
    write_csv_safe(detail_path, detail_fields, detail_rows)

    inside_count = sum(1 for row in detail_rows if row["inside_image"])
    outside_count = len(detail_rows) - inside_count
    # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "firms_csv": str(args.firms_csv),
        "target_date": args.target_date or None,
        "target_image": str(target_image_path) if target_image_path else None,
        "admin_geojson": str(args.admin_geojson) if args.admin_geojson else None,
        "probability_npy": str(args.probability_npy) if args.probability_npy else None,
        "threshold": args.threshold,
        "km_per_pixel": round(km_px, 6),
        "max_search_radius_px": max_radius_px,
        "total_firms_rows": len(firms_rows),
        "detail_rows_written": len(detail_rows),
        "points_inside_image": inside_count,
        "points_outside_image": outside_count,
        "admin_features_loaded": len(admin_features),
        "prediction_used": probability is not None,
    }
    if probability is not None and prediction_mask is not None:
        # Jumlah piksel positif dihitung untuk menunjukkan kelangkaan hotspot pada laporan.
        positive_pixels = int(np.count_nonzero(prediction_mask))
        summary["prediction_positive_pixels"] = positive_pixels
        summary["prediction_positive_ratio"] = round(positive_pixels / prediction_mask.size, 8)
        distance_summary: dict[str, dict[str, Any]] = {}
        for distance in distances:
            field = f"match_{format_distance(distance)}km"
            matched = sum(1 for row in detail_rows if row.get(field) is True)
            distance_summary[format_distance(distance)] = {
                "radius_km": distance,
                "matched_points": matched,
                "evaluated_points": inside_count,
                "hit_rate": round(matched / max(inside_count, 1), 6),
            }
        summary["distance_summary"] = distance_summary

    # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if any(row.get("kabupaten") for row in detail_rows):
        kabupaten_rows = summarize_by(detail_rows, "kabupaten", distances)
        write_csv_safe(kabupaten_path, list(kabupaten_rows[0].keys()) if kabupaten_rows else ["kabupaten"], kabupaten_rows)
    if any(row.get("kecamatan") for row in detail_rows):
        kecamatan_rows = summarize_by(detail_rows, "kecamatan", distances)
        write_csv_safe(kecamatan_path, list(kecamatan_rows[0].keys()) if kecamatan_rows else ["kecamatan"], kecamatan_rows)

    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    draw_overlay(
        overlay_path,
        base_image,
        probability,
        detail_rows,
        admin_features,
        args,
        distance_for_color=distances[min(1, len(distances) - 1)],
    )

    output_files = {
        "Detail titik": str(detail_path),
        "Ringkasan JSON": str(summary_path),
        "Overlay": str(overlay_path),
    }
    if kabupaten_path.exists():
        output_files["Rekap kabupaten"] = str(kabupaten_path)
    if kecamatan_path.exists():
        output_files["Rekap kecamatan"] = str(kecamatan_path)

    build_markdown_report(report_path, args, summary, output_files)
    output_files["Laporan Markdown"] = str(report_path)

    print("[geospatial] Selesai")
    print(f"[geospatial] Titik FIRMS: {len(firms_rows)}")
    print(f"[geospatial] Titik dalam gambar: {inside_count}")
    if probability is not None:
        for distance, stats in summary["distance_summary"].items():
            print(
                f"[geospatial] Radius {distance} km: "
                f"{stats['matched_points']}/{stats['evaluated_points']} "
                f"(hit_rate={stats['hit_rate']})"
            )
    for label, file_path in output_files.items():
        print(f"[geospatial] {label}: {Path(file_path).resolve()}")


if __name__ == "__main__":
    main()
