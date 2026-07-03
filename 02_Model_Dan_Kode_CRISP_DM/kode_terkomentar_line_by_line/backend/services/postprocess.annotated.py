# File anotasi dari `backend/services/postprocess.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Postprocessing backend: menyimpan probability map, heatmap, overlay, binary mask, dan garis batas kabupaten/kota.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from functools import lru_cache
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import json
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import math

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image, ImageDraw

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend import settings


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
WEB_MERCATOR_MAX_LAT = 85.05112878
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
WEB_MERCATOR_TILE_SIZE = 256.0


# Membuat langkah kerja bernama `save_probability_array`.
def save_probability_array(probability: np.ndarray, path: Path) -> None:
    # Mengolah angka piksel, mask hotspot, atau peta probabilitas.
    np.save(path, probability.astype(np.float32))


# Membuat langkah kerja bernama `_to_heatmap_rgb`.
def _to_heatmap_rgb(probability: np.ndarray) -> np.ndarray:
    # Menyimpan nilai ke `p` untuk dipakai pada langkah berikutnya.
    p = np.clip(probability, 0.0, 1.0)
    # Menyimpan nilai ke `r` untuk dipakai pada langkah berikutnya.
    r = (255.0 * p).astype(np.uint8)
    # Menyimpan nilai ke `g` untuk dipakai pada langkah berikutnya.
    g = (255.0 * (1.0 - np.abs((2.0 * p) - 1.0))).astype(np.uint8)
    # Menyimpan nilai ke `b` untuk dipakai pada langkah berikutnya.
    b = (255.0 * (1.0 - p)).astype(np.uint8)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.stack([r, g, b], axis=-1)


# Membuat langkah kerja bernama `save_heatmap`.
def save_heatmap(probability: np.ndarray, path: Path) -> None:
    # Menyimpan nilai ke `heatmap_rgb` untuk dipakai pada langkah berikutnya.
    heatmap_rgb = _to_heatmap_rgb(probability)
    # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
    image = Image.fromarray(heatmap_rgb)
    # Melanjutkan langkah kerja pada bagian kode ini.
    image.save(path)


# Membuat langkah kerja bernama `save_binary_mask`.
def save_binary_mask(probability: np.ndarray, threshold: float, path: Path) -> None:
    # Menggabungkan hasil deteksi merah menjadi mask hotspot.
    mask = np.where(probability >= threshold, 255, 0).astype(np.uint8)
    # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
    image = Image.fromarray(mask)
    # Melanjutkan langkah kerja pada bagian kode ini.
    image.save(path)


# Membuat langkah kerja bernama `_to_overlay_rgba`.
def _to_overlay_rgba(probability: np.ndarray, min_visible_probability: float) -> np.ndarray:
    # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
    probability = np.clip(probability, 0.0, 1.0)
    # Menyimpan nilai ke `visible` untuk dipakai pada langkah berikutnya.
    visible = np.clip(
        # Melanjutkan langkah kerja pada bagian kode ini.
        (probability - min_visible_probability) / max(1e-6, 1.0 - min_visible_probability),
        # Melanjutkan langkah kerja pada bagian kode ini.
        0.0,
        # Melanjutkan langkah kerja pada bagian kode ini.
        1.0,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Catatan asli dari pembuat kode.
    # Overlay dibuat khusus untuk foreground hotspot.
    # Catatan asli dari pembuat kode.
    # Nilai rendah dibuat transparan, nilai tinggi berwarna magenta/fuchsia
    # Catatan asli dari pembuat kode.
    # agar kontras dengan warna peta dasar Riau yang dominan hijau-biru-cokelat.
    # Menyimpan nilai ke `r` untuk dipakai pada langkah berikutnya.
    r = np.full_like(visible, 255.0)
    # Menyimpan nilai ke `g` untuk dipakai pada langkah berikutnya.
    g = 70.0 * (1.0 - visible)
    # Menyimpan nilai ke `b` untuk dipakai pada langkah berikutnya.
    b = 190.0 + (65.0 * visible)
    # Menyimpan nilai ke `a` untuk dipakai pada langkah berikutnya.
    a = 235.0 * np.power(visible, 0.85)

    # Menyimpan nilai ke `rgba` untuk dipakai pada langkah berikutnya.
    rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
    # Menyimpan nilai ke `rgba[visible <` untuk dipakai pada langkah berikutnya.
    rgba[visible <= 0.0] = np.array([0, 0, 0, 0], dtype=np.uint8)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return rgba


# Membuat langkah kerja bernama `_web_mercator_global_pixel`.
def _web_mercator_global_pixel(lon: float, lat: float, zoom: float) -> tuple[float, float]:
    # Menyimpan nilai ke `lat` untuk dipakai pada langkah berikutnya.
    lat = max(min(lat, WEB_MERCATOR_MAX_LAT), -WEB_MERCATOR_MAX_LAT)
    # Menyimpan nilai ke `scale` untuk dipakai pada langkah berikutnya.
    scale = WEB_MERCATOR_TILE_SIZE * (2.0**zoom)
    # Menyimpan nilai ke `x` untuk dipakai pada langkah berikutnya.
    x = (lon + 180.0) / 360.0 * scale
    # Menyimpan nilai ke `sin_lat` untuk dipakai pada langkah berikutnya.
    sin_lat = math.sin(math.radians(lat))
    # Menyimpan nilai ke `y` untuk dipakai pada langkah berikutnya.
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return x, y


# Membuat langkah kerja bernama `_lonlat_to_image_pixel`.
def _lonlat_to_image_pixel(lon: float, lat: float, width: int, height: int) -> tuple[float, float]:
    # Menyimpan nilai ke `center_x, center_y` untuk dipakai pada langkah berikutnya.
    center_x, center_y = _web_mercator_global_pixel(
        # Melanjutkan langkah kerja pada bagian kode ini.
        settings.ADMIN_BOUNDARY_CENTER_LON,
        # Melanjutkan langkah kerja pada bagian kode ini.
        settings.ADMIN_BOUNDARY_CENTER_LAT,
        # Melanjutkan langkah kerja pada bagian kode ini.
        settings.ADMIN_BOUNDARY_ZOOM,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `x, y` untuk dipakai pada langkah berikutnya.
    x, y = _web_mercator_global_pixel(lon, lat, settings.ADMIN_BOUNDARY_ZOOM)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return x - center_x + width / 2.0, y - center_y + height / 2.0


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@lru_cache(maxsize=1)
# Membuat langkah kerja bernama `_load_admin_boundary_geometries`.
def _load_admin_boundary_geometries() -> tuple[dict, ...]:
    # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
    path = settings.ADMIN_BOUNDARY_GEOJSON_PATH
    # Mengecek syarat sebelum melanjutkan proses.
    if not settings.ADMIN_BOUNDARY_OVERLAY_ENABLED or not path.exists():
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return tuple()

    # Mencoba menjalankan proses yang mungkin gagal.
    try:
        # Menyimpan nilai ke `data` untuk dipakai pada langkah berikutnya.
        data = json.loads(path.read_text(encoding="utf-8"))
    # Menangani kesalahan agar program tidak langsung berhenti.
    except (OSError, json.JSONDecodeError):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return tuple()

    # Menyimpan nilai ke `geometries` untuk dipakai pada langkah berikutnya.
    geometries: list[dict] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for feature in data.get("features", []):
        # Menyimpan nilai ke `geometry` untuk dipakai pada langkah berikutnya.
        geometry = feature.get("geometry") or {}
        # Mengecek syarat sebelum melanjutkan proses.
        if geometry.get("type") in {"Polygon", "MultiPolygon"}:
            # Melanjutkan langkah kerja pada bagian kode ini.
            geometries.append(geometry)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tuple(geometries)


# Membuat langkah kerja bernama `_geometry_to_pixel_paths`.
def _geometry_to_pixel_paths(geometry: dict, width: int, height: int) -> list[list[tuple[float, float]]]:
    # Menyimpan nilai ke `geom_type` untuk dipakai pada langkah berikutnya.
    geom_type = geometry.get("type")
    # Menyimpan nilai ke `coordinates` untuk dipakai pada langkah berikutnya.
    coordinates = geometry.get("coordinates", [])
    # Menyimpan nilai ke `polygons` untuk dipakai pada langkah berikutnya.
    polygons = [coordinates] if geom_type == "Polygon" else coordinates if geom_type == "MultiPolygon" else []

    # Menyimpan nilai ke `paths` untuk dipakai pada langkah berikutnya.
    paths: list[list[tuple[float, float]]] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for polygon in polygons:
        # Mengecek syarat sebelum melanjutkan proses.
        if not polygon:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Menyimpan nilai ke `outer_ring` untuk dipakai pada langkah berikutnya.
        outer_ring = polygon[0]
        # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
        path = []
        # Mengulang proses untuk setiap data dalam daftar.
        for coordinate in outer_ring:
            # Mengecek syarat sebelum melanjutkan proses.
            if len(coordinate) < 2:
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Menyimpan nilai ke `lon, lat` untuk dipakai pada langkah berikutnya.
            lon, lat = float(coordinate[0]), float(coordinate[1])
            # Menyimpan nilai ke `path.append(_lonlat_to_image_pixel(lon` untuk dipakai pada langkah berikutnya.
            path.append(_lonlat_to_image_pixel(lon=lon, lat=lat, width=width, height=height))
        # Mengecek syarat sebelum melanjutkan proses.
        if len(path) >= 2:
            # Melanjutkan langkah kerja pada bagian kode ini.
            paths.append(path)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return paths


# Membuat langkah kerja bernama `_draw_admin_boundaries`.
def _draw_admin_boundaries(image: Image.Image) -> Image.Image:
    # Menyimpan nilai ke `geometries` untuk dipakai pada langkah berikutnya.
    geometries = _load_admin_boundary_geometries()
    # Mengecek syarat sebelum melanjutkan proses.
    if not geometries:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return image

    # Menyimpan nilai ke `width, height` untuk dipakai pada langkah berikutnya.
    width, height = image.size
    # Menyimpan nilai ke `draw` untuk dipakai pada langkah berikutnya.
    draw = ImageDraw.Draw(image)

    # Catatan asli dari pembuat kode.
    # Dua layer garis membuat batas tetap terlihat di atas peta satelit dan heatmap:
    # Catatan asli dari pembuat kode.
    # outline gelap lebih tebal, lalu garis cyan terang di atasnya.
    # Menyimpan nilai ke `outline_width` untuk dipakai pada langkah berikutnya.
    outline_width = max(settings.ADMIN_BOUNDARY_OUTLINE_WIDTH, settings.ADMIN_BOUNDARY_LINE_WIDTH)
    # Menyimpan nilai ke `line_width` untuk dipakai pada langkah berikutnya.
    line_width = max(1, settings.ADMIN_BOUNDARY_LINE_WIDTH)
    # Mengulang proses untuk setiap data dalam daftar.
    for geometry in geometries:
        # Mengulang proses untuk setiap data dalam daftar.
        for path in _geometry_to_pixel_paths(geometry=geometry, width=width, height=height):
            # Menyimpan nilai ke `draw.line(path, fill` untuk dipakai pada langkah berikutnya.
            draw.line(path, fill=(5, 18, 28, 220), width=outline_width, joint="curve")
            # Menyimpan nilai ke `draw.line(path, fill` untuk dipakai pada langkah berikutnya.
            draw.line(path, fill=(0, 245, 255, 235), width=line_width, joint="curve")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return image


# Membuat langkah kerja bernama `save_overlay`.
def save_overlay(
    # Menjelaskan data `probability` yang disimpan atau dikirim pada bagian ini.
    probability: np.ndarray,
    # Menjelaskan data `reference_input` yang disimpan atau dikirim pada bagian ini.
    reference_input: Path,
    # Menjelaskan data `destination_path` yang disimpan atau dikirim pada bagian ini.
    destination_path: Path,
    # Menyimpan nilai ke `base_map_path` untuk dipakai pada langkah berikutnya.
    base_map_path: Path | None = None,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> None:
    # Mengecek syarat sebelum melanjutkan proses.
    if base_map_path and base_map_path.exists():
        # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
        with Image.open(base_map_path) as base:
            # Menyimpan nilai ke `base_image` untuk dipakai pada langkah berikutnya.
            base_image = base.convert("RGB")
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
        with Image.open(reference_input) as ref:
            # Menyimpan nilai ke `base_image` untuk dipakai pada langkah berikutnya.
            base_image = ref.convert("RGB")

    # Menyimpan nilai ke `h, w` untuk dipakai pada langkah berikutnya.
    h, w = probability.shape
    # Menyimpan nilai ke `base_image` untuk dipakai pada langkah berikutnya.
    base_image = base_image.resize((w, h), Image.BILINEAR)

    # Menyimpan nilai ke `overlay_rgba` untuk dipakai pada langkah berikutnya.
    overlay_rgba = _to_overlay_rgba(
        # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
        probability=probability,
        # Menyimpan nilai ke `min_visible_probability` untuk dipakai pada langkah berikutnya.
        min_visible_probability=settings.OVERLAY_MIN_VISIBLE_PROBABILITY,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `overlay_image` untuk dipakai pada langkah berikutnya.
    overlay_image = Image.fromarray(overlay_rgba)

    # Menyimpan nilai ke `merged` untuk dipakai pada langkah berikutnya.
    merged = Image.alpha_composite(base_image.convert("RGBA"), overlay_image)
    # Menyimpan nilai ke `merged` untuk dipakai pada langkah berikutnya.
    merged = _draw_admin_boundaries(merged)
    # Melanjutkan langkah kerja pada bagian kode ini.
    merged.save(destination_path)
