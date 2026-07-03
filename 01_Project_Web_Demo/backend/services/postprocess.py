"""Komentar file skripsi:
Postprocessing backend untuk membuat probability file, heatmap, binary mask, overlay peta,
batas kabupaten/kota Riau, legenda wilayah, dan visualisasi hasil prediksi hotspot.

Konteks laporan: file ini mendukung BAB IV pada bagian evaluation dan deployment,
karena mengubah output numerik model ConvLSTM menjadi artefak visual yang dapat dibaca pengguna.
"""

# Path digunakan untuk menerima lokasi file output visual seperti .npy, .png, dan peta dasar.
from pathlib import Path

# lru_cache digunakan untuk menyimpan hasil pembacaan data/font agar tidak diproses berulang.
from functools import lru_cache

# json digunakan untuk membaca file GeoJSON batas kabupaten/kota Riau.
import json

# math digunakan untuk perhitungan proyeksi Web Mercator dan ukuran legenda.
import math

# NumPy digunakan untuk mengolah probability map hasil prediksi model.
import numpy as np

# PIL digunakan untuk membuat, menggambar, menggabungkan, dan menyimpan gambar visualisasi.
from PIL import Image, ImageDraw, ImageFont

# settings berisi konfigurasi overlay, peta dasar, path GeoJSON, threshold tampilan, dan parameter visual.
from backend import settings


# Batas latitude maksimum pada proyeksi Web Mercator.
WEB_MERCATOR_MAX_LAT = 85.05112878

# Ukuran tile standar Web Mercator dalam piksel.
WEB_MERCATOR_TILE_SIZE = 256.0

# Palet warna cadangan untuk membedakan wilayah administrasi jika nama wilayah tidak ada pada mapping khusus.
ADMIN_REGION_PALETTE = (
    (0, 180, 216),
    (255, 214, 10),
    (255, 0, 110),
    (131, 56, 236),
    (6, 214, 160),
    (255, 122, 0),
    (139, 69, 19),
    (124, 181, 24),
    (241, 91, 181),
    (94, 96, 206),
    (17, 138, 178),
    (239, 71, 111),
)

# Mapping warna khusus untuk kabupaten/kota Riau agar warna konsisten pada setiap hasil overlay.
ADMIN_REGION_COLORS_BY_NAME = {
    "Kota Pekanbaru": (139, 69, 19),
    "Kota Dumai": (255, 122, 0),
    "Siak": (239, 71, 111),
    "Bengkalis": (0, 180, 216),
    "Kampar": (131, 56, 236),
    "Rokan Hilir": (94, 96, 206),
    "Rokan Hulu": (17, 138, 178),
    "Kuantan Singingi": (124, 181, 24),
    "Pelalawan": (241, 91, 181),
    "Indragiri Hulu": (255, 0, 110),
    "Indragiri Hilir": (255, 214, 10),
    "Kepulauan Meranti": (6, 214, 160),
}

# Mapping nomor wilayah untuk memudahkan pembacaan overlay kabupaten/kota pada dashboard.
ADMIN_REGION_NUMBERS_BY_NAME = {
    "Kota Pekanbaru": 1,
    "Kota Dumai": 2,
    "Siak": 3,
    "Bengkalis": 4,
    "Kampar": 5,
    "Rokan Hilir": 6,
    "Rokan Hulu": 7,
    "Kuantan Singingi": 8,
    "Pelalawan": 9,
    "Indragiri Hulu": 10,
    "Indragiri Hilir": 11,
    "Kepulauan Meranti": 12,
}


# Fungsi untuk menyimpan probability map hasil model ke file .npy.
def save_probability_array(probability: np.ndarray, path: Path) -> None:
    # Menyimpan array probabilitas sebagai float32 agar ukuran file efisien dan tetap presisi.
    np.save(path, probability.astype(np.float32))


# Fungsi internal untuk mengubah probability map menjadi citra heatmap RGB.
def _to_heatmap_rgb(probability: np.ndarray) -> np.ndarray:
    # Membatasi nilai probabilitas pada rentang 0 sampai 1 agar visualisasi stabil.
    p = np.clip(probability, 0.0, 1.0)

    # Channel merah makin tinggi ketika probabilitas risiko makin besar.
    r = (255.0 * p).astype(np.uint8)

    # Channel hijau tinggi pada nilai tengah sehingga transisi warna terlihat jelas.
    g = (255.0 * (1.0 - np.abs((2.0 * p) - 1.0))).astype(np.uint8)

    # Channel biru makin tinggi ketika probabilitas risiko makin rendah.
    b = (255.0 * (1.0 - p)).astype(np.uint8)

    # Menggabungkan channel R, G, B menjadi citra heatmap.
    return np.stack([r, g, b], axis=-1)


# Fungsi untuk menyimpan heatmap prediksi ke file gambar.
def save_heatmap(probability: np.ndarray, path: Path) -> None:
    # Mengubah probability map menjadi array RGB.
    heatmap_rgb = _to_heatmap_rgb(probability)

    # Mengubah array RGB menjadi object gambar PIL.
    image = Image.fromarray(heatmap_rgb)

    # Menyimpan heatmap ke path output.
    image.save(path)


# Fungsi untuk menyimpan binary mask berdasarkan threshold yang diberikan pengguna.
def save_binary_mask(probability: np.ndarray, threshold: float, path: Path) -> None:
    # Piksel bernilai 255 jika probabilitas >= threshold, selain itu 0.
    mask = np.where(probability >= threshold, 255, 0).astype(np.uint8)

    # Mengubah array mask menjadi gambar grayscale.
    image = Image.fromarray(mask)

    # Menyimpan binary mask ke path output.
    image.save(path)


# Fungsi internal untuk mengubah probability map menjadi overlay RGBA transparan.
def _to_overlay_rgba(probability: np.ndarray, min_visible_probability: float) -> np.ndarray:
    # Membatasi nilai probability map pada rentang valid 0 sampai 1.
    probability = np.clip(probability, 0.0, 1.0)

    # Menghitung intensitas tampilan berdasarkan probabilitas minimum yang terlihat.
    visible = np.clip(
        (probability - min_visible_probability) / max(1e-6, 1.0 - min_visible_probability),
        0.0,
        1.0,
    )

    # Nilai rendah dibuat lebih transparan agar peta tidak tertutup warna.
    # Nilai tinggi tetap dibuat terang supaya titik/area risiko lebih menonjol.
    # Channel merah dibuat konstan tinggi untuk menonjolkan area risiko.
    r = np.full_like(visible, 255.0)

    # Channel hijau dinaikkan mengikuti intensitas visible.
    g = 30.0 + (110.0 * visible)

    # Channel biru dinaikkan mengikuti intensitas visible untuk memberi warna magenta/merah muda.
    b = 180.0 + (75.0 * visible)

    # Alpha menentukan transparansi; probabilitas rendah menjadi lebih transparan.
    a = 240.0 * np.power(visible, 1.65)

    # Menggabungkan channel R, G, B, A menjadi gambar overlay.
    rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)

    # Piksel di bawah batas visible dibuat transparan penuh.
    rgba[visible <= 0.0] = np.array([0, 0, 0, 0], dtype=np.uint8)

    # Mengembalikan overlay RGBA.
    return rgba


# Fungsi internal untuk mengubah koordinat lon/lat menjadi posisi piksel global Web Mercator.
def _web_mercator_global_pixel(lon: float, lat: float, zoom: float) -> tuple[float, float]:
    # Membatasi latitude agar tetap berada pada rentang valid Web Mercator.
    lat = max(min(lat, WEB_MERCATOR_MAX_LAT), -WEB_MERCATOR_MAX_LAT)

    # Menghitung skala peta berdasarkan zoom.
    scale = WEB_MERCATOR_TILE_SIZE * (2.0**zoom)

    # Mengubah longitude menjadi koordinat X global.
    x = (lon + 180.0) / 360.0 * scale

    # Menghitung nilai sinus latitude untuk rumus Web Mercator.
    sin_lat = math.sin(math.radians(lat))

    # Mengubah latitude menjadi koordinat Y global.
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale

    # Mengembalikan koordinat piksel global.
    return x, y


# Fungsi internal untuk mengubah lon/lat GeoJSON menjadi piksel pada gambar overlay.
def _lonlat_to_image_pixel(lon: float, lat: float, width: int, height: int) -> tuple[float, float]:
    # Menghitung titik pusat peta berdasarkan konfigurasi longitude, latitude, dan zoom.
    center_x, center_y = _web_mercator_global_pixel(
        settings.ADMIN_BOUNDARY_CENTER_LON,
        settings.ADMIN_BOUNDARY_CENTER_LAT,
        settings.ADMIN_BOUNDARY_ZOOM,
    )

    # Menghitung koordinat global titik wilayah.
    x, y = _web_mercator_global_pixel(lon, lat, settings.ADMIN_BOUNDARY_ZOOM)

    # Mengubah koordinat global menjadi koordinat lokal gambar.
    return x - center_x + width / 2.0, y - center_y + height / 2.0


# Fungsi internal untuk mengambil nama wilayah dari property GeoJSON.
def _admin_feature_name(feature: dict) -> str:
    # Mengambil bagian properties dari feature GeoJSON.
    properties = feature.get("properties") or {}

    # Mencoba beberapa kemungkinan nama kolom wilayah.
    for key in ("kabupaten_kota", "shapeName", "name", "NAME_2", "NAMOBJ"):
        # Mengambil nilai nama berdasarkan key.
        value = properties.get(key)

        # Jika nilai ditemukan, gunakan sebagai nama wilayah.
        if value:
            return str(value)

    # Jika tidak ada nama wilayah, gunakan label umum.
    return "Wilayah"


# Cache dipakai agar file GeoJSON batas administrasi tidak dibaca berulang-ulang.
@lru_cache(maxsize=1)
def _load_admin_boundary_features() -> tuple[dict, ...]:
    # Mengambil path GeoJSON batas kabupaten/kota dari settings.
    path = settings.ADMIN_BOUNDARY_GEOJSON_PATH

    # Jika overlay dimatikan atau file tidak ada, kembalikan tuple kosong.
    if not settings.ADMIN_BOUNDARY_OVERLAY_ENABLED or not path.exists():
        return tuple()

    # Mencoba membaca file GeoJSON.
    try:
        # Membaca teks GeoJSON dan mengubahnya menjadi dictionary.
        data = json.loads(path.read_text(encoding="utf-8"))

    # Jika file gagal dibaca atau JSON tidak valid, overlay batas wilayah dikosongkan.
    except (OSError, json.JSONDecodeError):
        return tuple()

    # Menyiapkan list feature wilayah yang valid.
    features: list[dict] = []

    # Membaca setiap feature dari GeoJSON.
    for feature in data.get("features", []):
        # Mengambil geometry feature.
        geometry = feature.get("geometry") or {}

        # Hanya geometry Polygon dan MultiPolygon yang digunakan sebagai batas wilayah.
        if geometry.get("type") in {"Polygon", "MultiPolygon"}:
            # Menyimpan nama dan geometry wilayah.
            features.append(
                {
                    "name": _admin_feature_name(feature),
                    "geometry": geometry,
                }
            )

    # Mengembalikan feature dalam bentuk tuple agar aman untuk cache.
    return tuple(features)


# Fungsi internal untuk mengambil warna wilayah berdasarkan nama atau palet cadangan.
def _admin_region_color(name: str, index: int, alpha: int = 255) -> tuple[int, int, int, int]:
    # Mengambil warna dari mapping nama, jika tidak ada memakai palet berdasarkan index.
    color = ADMIN_REGION_COLORS_BY_NAME.get(name, ADMIN_REGION_PALETTE[index % len(ADMIN_REGION_PALETTE)])

    # Mengembalikan warna RGBA dengan alpha.
    return (*color, alpha)


# Fungsi internal untuk mengambil nomor wilayah berdasarkan nama atau fallback index.
def _admin_region_number(name: str, fallback: int) -> int:
    # Mengambil nomor wilayah dari mapping, jika tidak ada memakai fallback.
    return ADMIN_REGION_NUMBERS_BY_NAME.get(name, fallback)


# Fungsi internal untuk mengubah warna RGB menjadi format hex untuk frontend.
def _color_to_hex(color: tuple[int, int, int]) -> str:
    # Mengubah tuple RGB menjadi string seperti #ff0000.
    return "#{:02x}{:02x}{:02x}".format(*color)


# Fungsi untuk membuat daftar legenda kabupaten/kota yang dikirim ke frontend.
def get_admin_region_legend() -> list[dict]:
    # Menyiapkan list legend.
    legend: list[dict] = []

    # Membaca seluruh feature batas administrasi yang aktif.
    for index, feature in enumerate(_load_admin_boundary_features()):
        # Mengambil nomor wilayah berdasarkan nama.
        number = _admin_region_number(feature["name"], index + 1)

        # Mengambil warna wilayah berdasarkan nama.
        color = ADMIN_REGION_COLORS_BY_NAME.get(
            feature["name"],
            ADMIN_REGION_PALETTE[index % len(ADMIN_REGION_PALETTE)],
        )

        # Menambahkan item legend ke list.
        legend.append(
            {
                "number": number,
                "name": feature["name"],
                "color": _color_to_hex(color),
            }
        )

    # Mengurutkan legenda berdasarkan nomor wilayah.
    return sorted(legend, key=lambda item: item["number"])


# Fungsi internal untuk mengubah geometry GeoJSON menjadi daftar path piksel.
def _geometry_to_pixel_paths(geometry: dict, width: int, height: int) -> list[list[tuple[float, float]]]:
    # Mengambil tipe geometry.
    geom_type = geometry.get("type")

    # Mengambil koordinat geometry.
    coordinates = geometry.get("coordinates", [])

    # Menyamakan struktur Polygon dan MultiPolygon menjadi list polygon.
    polygons = [coordinates] if geom_type == "Polygon" else coordinates if geom_type == "MultiPolygon" else []

    # Menyiapkan list path piksel.
    paths: list[list[tuple[float, float]]] = []

    # Membaca setiap polygon.
    for polygon in polygons:
        # Melewati polygon kosong.
        if not polygon:
            continue

        # Mengambil ring luar polygon sebagai batas wilayah.
        outer_ring = polygon[0]

        # Menyiapkan path piksel untuk satu polygon.
        path = []

        # Membaca setiap koordinat lon/lat dalam ring luar.
        for coordinate in outer_ring:
            # Melewati koordinat yang tidak memiliki lon dan lat.
            if len(coordinate) < 2:
                continue

            # Mengambil longitude dan latitude.
            lon, lat = float(coordinate[0]), float(coordinate[1])

            # Mengubah lon/lat menjadi piksel gambar.
            path.append(_lonlat_to_image_pixel(lon=lon, lat=lat, width=width, height=height))

        # Path minimal harus memiliki dua titik agar dapat digambar sebagai garis.
        if len(path) >= 2:
            paths.append(path)

    # Mengembalikan seluruh path piksel.
    return paths


# Fungsi internal untuk menghitung batas kotak dari sebuah path.
def _path_bounds(path: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    # Mengambil seluruh koordinat X.
    xs = [point[0] for point in path]

    # Mengambil seluruh koordinat Y.
    ys = [point[1] for point in path]

    # Mengembalikan batas minimum dan maksimum path.
    return min(xs), min(ys), max(xs), max(ys)


# Fungsi internal untuk menghitung luas bounding box path.
def _path_bbox_area(path: list[tuple[float, float]]) -> float:
    # Mengambil batas path.
    min_x, min_y, max_x, max_y = _path_bounds(path)

    # Menghitung luas bounding box dengan proteksi nilai negatif.
    return max(0.0, max_x - min_x) * max(0.0, max_y - min_y)


# Cache font dipakai agar loading font tidak dilakukan berulang.
@lru_cache(maxsize=8)
def _load_label_font(size: int) -> ImageFont.ImageFont:
    # Daftar kandidat font yang dicoba pada Windows dan lingkungan umum.
    candidates = (
        "arialbd.ttf",
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )

    # Mencoba setiap kandidat font.
    for candidate in candidates:
        # Mencoba load font TrueType.
        try:
            return ImageFont.truetype(candidate, size=size)

        # Jika font tidak ditemukan, lanjut ke kandidat berikutnya.
        except OSError:
            continue

    # Jika semua font gagal, gunakan font default PIL.
    return ImageFont.load_default()


# Fungsi internal untuk memecah nama wilayah menjadi beberapa baris agar muat pada peta.
def _wrap_region_label(name: str) -> str:
    # Mengubah nama menjadi uppercase dan menghapus kata KABUPATEN/KOTA agar label lebih pendek.
    words = name.upper().replace("KABUPATEN ", "").replace("KOTA ", "").split()

    # Jika nama kosong, gunakan label umum.
    if not words:
        return "WILAYAH"

    # Menyiapkan list baris label.
    lines: list[str] = []

    # Menyiapkan teks baris sementara.
    current = ""

    # Menyusun kata menjadi baris dengan panjang maksimal sekitar 14 karakter.
    for word in words:
        # Membentuk kandidat baris.
        candidate = f"{current} {word}".strip()

        # Jika panjang masih cukup, kata dimasukkan ke baris saat ini.
        if len(candidate) <= 14:
            current = candidate

        # Jika terlalu panjang, baris lama disimpan dan kata baru menjadi baris berikutnya.
        else:
            if current:
                lines.append(current)
            current = word

    # Menambahkan sisa baris terakhir jika ada.
    if current:
        lines.append(current)

    # Mengembalikan maksimal tiga baris label.
    return "\n".join(lines[:3])


# Fungsi internal untuk menggambar label nama wilayah administrasi pada overlay.
def _draw_admin_region_label(
    # Object draw PIL untuk menggambar teks.
    draw: ImageDraw.ImageDraw,
    # Nama wilayah administrasi.
    name: str,
    # Daftar path piksel wilayah.
    paths: list[list[tuple[float, float]]],
    # Lebar gambar overlay.
    width: int,
    # Tinggi gambar overlay.
    height: int,
    # Daftar kotak label yang sudah terpakai agar label tidak saling tumpang tindih.
    occupied_boxes: list[tuple[float, float, float, float]],
) -> None:
    # Jika label dimatikan atau path kosong, fungsi berhenti.
    if not settings.ADMIN_REGION_LABEL_ENABLED or not paths:
        return

    # Mengambil path terbesar sebagai lokasi label paling representatif.
    largest_path = max(paths, key=_path_bbox_area)

    # Jika wilayah terlalu kecil, label tidak digambar agar peta tidak penuh.
    if _path_bbox_area(largest_path) < settings.ADMIN_REGION_LABEL_MIN_AREA:
        return

    # Mengambil batas path terbesar.
    min_x, min_y, max_x, max_y = _path_bounds(largest_path)

    # Menghitung titik tengah X label dan membatasinya agar tetap berada dalam gambar.
    center_x = min(max((min_x + max_x) / 2.0, 12.0), width - 12.0)

    # Menghitung titik tengah Y label dan membatasinya agar tetap berada dalam gambar.
    center_y = min(max((min_y + max_y) / 2.0, 12.0), height - 12.0)

    # Memuat font label wilayah.
    font = _load_label_font(settings.ADMIN_REGION_LABEL_FONT_SIZE)

    # Membuat teks label yang sudah di-wrap.
    label = _wrap_region_label(name)

    # Menghitung bounding box teks label.
    bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=2, stroke_width=2)

    # Menghitung lebar teks.
    text_w = bbox[2] - bbox[0]

    # Menghitung tinggi teks.
    text_h = bbox[3] - bbox[1]

    # Daftar opsi offset posisi label untuk menghindari tumpang tindih.
    offsets = (
        (0.0, 0.0),
        (0.0, -34.0),
        (0.0, 34.0),
        (-72.0, 0.0),
        (72.0, 0.0),
        (-72.0, -34.0),
        (72.0, -34.0),
        (-72.0, 34.0),
        (72.0, 34.0),
    )

    # Posisi teks terpilih, default None.
    selected_xy: tuple[float, float] | None = None

    # Bounding box label terpilih, default None.
    selected_box: tuple[float, float, float, float] | None = None

    # Mencoba setiap offset untuk mencari posisi label yang tidak overlap.
    for dx, dy in offsets:
        # Menghitung posisi X label.
        x = min(max(center_x - text_w / 2.0 + dx, 8.0), width - text_w - 8.0)

        # Menghitung posisi Y label.
        y = min(max(center_y - text_h / 2.0 + dy, 8.0), height - text_h - 8.0)

        # Membuat bounding box kandidat label.
        candidate_box = (x - 4.0, y - 4.0, x + text_w + 4.0, y + text_h + 4.0)

        # Mengecek apakah kandidat label overlap dengan label lain.
        overlaps = any(
            not (
                candidate_box[2] < box[0]
                or candidate_box[0] > box[2]
                or candidate_box[3] < box[1]
                or candidate_box[1] > box[3]
            )
            for box in occupied_boxes
        )

        # Jika tidak overlap, posisi ini dipilih.
        if not overlaps:
            selected_xy = (x, y)
            selected_box = candidate_box
            break

    # Jika semua posisi overlap, gunakan posisi tengah yang dibatasi area gambar.
    if selected_xy is None:
        # Menghitung fallback posisi label.
        selected_xy = (
            min(max(center_x - text_w / 2.0, 8.0), width - text_w - 8.0),
            min(max(center_y - text_h / 2.0, 8.0), height - text_h - 8.0),
        )

        # Membuat bounding box fallback.
        selected_box = (
            selected_xy[0] - 4.0,
            selected_xy[1] - 4.0,
            selected_xy[0] + text_w + 4.0,
            selected_xy[1] + text_h + 4.0,
        )

    # Menggambar teks label wilayah dengan outline gelap agar mudah terbaca.
    draw.multiline_text(
        selected_xy,
        label,
        fill=(255, 255, 255, 245),
        font=font,
        spacing=2,
        align="center",
        stroke_width=3,
        stroke_fill=(0, 0, 0, 230),
    )

    # Menyimpan bounding box label agar label berikutnya tidak tumpang tindih.
    occupied_boxes.append(selected_box)


# Fungsi internal untuk menggambar nomor wilayah administrasi pada overlay.
def _draw_admin_region_number(
    # Object draw PIL.
    draw: ImageDraw.ImageDraw,
    # Nama wilayah administrasi.
    name: str,
    # Daftar path piksel wilayah.
    paths: list[list[tuple[float, float]]],
    # Lebar gambar.
    width: int,
    # Tinggi gambar.
    height: int,
    # Index wilayah sebagai fallback nomor/warna.
    index: int,
) -> None:
    # Jika path kosong, nomor tidak dapat digambar.
    if not paths:
        return

    # Mengambil path terbesar sebagai lokasi nomor.
    largest_path = max(paths, key=_path_bbox_area)

    # Jika wilayah terlalu kecil, nomor tidak digambar.
    if _path_bbox_area(largest_path) < settings.ADMIN_REGION_LABEL_MIN_AREA:
        return

    # Mengambil batas path terbesar.
    min_x, min_y, max_x, max_y = _path_bounds(largest_path)

    # Menghitung posisi tengah X nomor.
    center_x = min(max((min_x + max_x) / 2.0, 16.0), width - 16.0)

    # Menghitung posisi tengah Y nomor.
    center_y = min(max((min_y + max_y) / 2.0, 16.0), height - 16.0)

    # Mengambil nomor wilayah berdasarkan nama.
    number = str(_admin_region_number(name, index + 1))

    # Memuat font untuk nomor wilayah.
    font = _load_label_font(max(14, settings.ADMIN_REGION_LABEL_FONT_SIZE - 6))

    # Menentukan radius lingkaran nomor; nomor dua digit dibuat sedikit lebih besar.
    radius = 15 if len(number) == 1 else 17

    # Mengambil warna wilayah dengan alpha tinggi.
    color = _admin_region_color(name, index, 235)

    # Menggambar lingkaran warna sebagai background nomor.
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=color,
        outline=(255, 255, 255, 245),
        width=3,
    )

    # Menggambar outline gelap di luar lingkaran agar terlihat pada berbagai background.
    draw.ellipse(
        (center_x - radius - 2, center_y - radius - 2, center_x + radius + 2, center_y + radius + 2),
        outline=(0, 0, 0, 210),
        width=2,
    )

    # Menghitung ukuran teks nomor.
    bbox = draw.textbbox((0, 0), number, font=font, stroke_width=1)

    # Menghitung lebar teks nomor.
    text_w = bbox[2] - bbox[0]

    # Menghitung tinggi teks nomor.
    text_h = bbox[3] - bbox[1]

    # Menggambar nomor di tengah lingkaran.
    draw.text(
        (center_x - text_w / 2.0, center_y - text_h / 2.0 - 1.0),
        number,
        fill=(255, 255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 220),
    )


# Fungsi internal untuk menggambar batas administrasi, nomor, dan label wilayah.
def _draw_admin_boundaries(
    # Gambar overlay yang akan diberi batas administrasi.
    image: Image.Image,
    # Opsi apakah label nama wilayah digambar.
    draw_labels: bool = False,
    # Opsi apakah nomor wilayah digambar.
    draw_numbers: bool = True,
) -> Image.Image:
    # Memuat feature batas kabupaten/kota dari GeoJSON.
    features = _load_admin_boundary_features()

    # Jika tidak ada feature, gambar dikembalikan tanpa perubahan.
    if not features:
        return image

    # Mengambil ukuran gambar.
    width, height = image.size

    # Membuat layer transparan untuk isi warna wilayah.
    fill_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Membuat object draw untuk layer isi wilayah.
    fill_draw = ImageDraw.Draw(fill_layer)

    # Jika fill wilayah aktif, gambar warna transparan per kabupaten/kota.
    if settings.ADMIN_REGION_FILL_ENABLED:
        # Membatasi alpha agar tidak terlalu menutup peta.
        alpha = max(0, min(120, settings.ADMIN_REGION_FILL_ALPHA))

        # Iterasi seluruh feature wilayah.
        for index, feature in enumerate(features):
            # Mengubah geometry wilayah menjadi path piksel.
            paths = _geometry_to_pixel_paths(
                geometry=feature["geometry"],
                width=width,
                height=height,
            )

            # Menggambar polygon isi wilayah.
            for path in paths:
                # Polygon minimal perlu tiga titik.
                if len(path) >= 3:
                    fill_draw.polygon(path, fill=_admin_region_color(feature["name"], index, alpha))

        # Menggabungkan layer isi wilayah ke gambar utama.
        image = Image.alpha_composite(image.convert("RGBA"), fill_layer)

    # Membuat object draw untuk gambar utama.
    draw = ImageDraw.Draw(image)

    # Garis batas dibuat netral: outline hitam dan garis putih.
    # Ini menjaga batas kabupaten/kota tetap terlihat tanpa terlihat seperti warna risiko.
    # Menentukan ketebalan outline batas wilayah.
    outline_width = max(settings.ADMIN_BOUNDARY_OUTLINE_WIDTH, settings.ADMIN_BOUNDARY_LINE_WIDTH)

    # Menentukan ketebalan garis utama batas wilayah.
    line_width = max(1, settings.ADMIN_BOUNDARY_LINE_WIDTH)

    # Menyiapkan list feature path untuk menggambar nomor/label setelah garis.
    feature_paths: list[tuple[int, str, list[list[tuple[float, float]]]]] = []

    # Iterasi setiap feature wilayah administrasi.
    for index, feature in enumerate(features):
        # Mengubah geometry wilayah menjadi path piksel.
        paths = _geometry_to_pixel_paths(
            geometry=feature["geometry"],
            width=width,
            height=height,
        )

        # Menyimpan path feature untuk proses nomor/label.
        feature_paths.append((index, feature["name"], paths))

        # Menggambar garis batas untuk setiap path wilayah.
        for path in paths:
            # Menggambar outline hitam.
            draw.line(path, fill=(0, 0, 0, 235), width=outline_width, joint="curve")

            # Menggambar garis putih di atas outline agar batas lebih jelas.
            draw.line(path, fill=(255, 255, 255, 230), width=line_width, joint="curve")

    # Jika nomor wilayah aktif, gambar nomor setiap wilayah.
    if draw_numbers:
        # Iterasi path wilayah untuk menggambar nomor.
        for index, name, paths in feature_paths:
            _draw_admin_region_number(
                draw=draw,
                name=name,
                paths=paths,
                width=width,
                height=height,
                index=index,
            )

    # Menyiapkan daftar area label yang sudah terpakai.
    occupied_boxes: list[tuple[float, float, float, float]] = []

    # Jika label nama wilayah aktif, gambar label nama wilayah.
    if draw_labels:
        # Iterasi path wilayah untuk menggambar label.
        for _, name, paths in feature_paths:
            _draw_admin_region_label(
                draw=draw,
                name=name,
                paths=paths,
                width=width,
                height=height,
                occupied_boxes=occupied_boxes,
            )

    # Mengembalikan gambar dengan batas administrasi.
    return image


# Fungsi internal untuk menambahkan legenda kabupaten/kota di bawah gambar.
def _append_admin_legend(image: Image.Image) -> Image.Image:
    # Mengambil daftar legenda wilayah.
    legend = get_admin_region_legend()

    # Jika legend kosong, gambar dikembalikan tanpa perubahan.
    if not legend:
        return image

    # Mengambil ukuran gambar.
    width, height = image.size

    # Memuat font judul legenda.
    title_font = _load_label_font(24)

    # Memuat font item legenda.
    item_font = _load_label_font(20)

    # Menentukan jumlah kolom legend berdasarkan lebar gambar.
    columns = 4 if width >= 1200 else 2

    # Menghitung jumlah baris legend.
    rows = math.ceil(len(legend) / columns)

    # Menghitung tinggi area legend tambahan.
    legend_height = 92 + rows * 42

    # Membuat kanvas output baru yang lebih tinggi untuk menampung legend.
    output = Image.new("RGBA", (width, height + legend_height), (248, 246, 238, 255))

    # Menempelkan gambar utama ke bagian atas output.
    output.alpha_composite(image.convert("RGBA"), (0, 0))

    # Membuat object draw untuk output.
    draw = ImageDraw.Draw(output)

    # Menentukan posisi atas area legend.
    top = height

    # Menggambar background area legend.
    draw.rectangle((0, top, width, height + legend_height), fill=(248, 246, 238, 255))

    # Menggambar garis pemisah antara gambar dan legend.
    draw.line((0, top, width, top), fill=(0, 0, 0, 80), width=2)

    # Menggambar judul legend.
    draw.text(
        (28, top + 18),
        "Legenda kabupaten/kota Riau",
        fill=(24, 49, 42, 255),
        font=title_font,
    )

    # Menggambar catatan bahwa warna wilayah bukan tingkat risiko.
    draw.text(
        (28, top + 50),
        "Warna wilayah hanya sebagai pembeda administrasi, bukan tingkat risiko hotspot.",
        fill=(95, 113, 107, 255),
        font=item_font,
    )

    # Menentukan posisi awal item legend.
    item_top = top + 86

    # Menghitung lebar setiap kolom legend.
    column_width = width / columns

    # Menggambar setiap item legend.
    for index, item in enumerate(legend):
        # Menghitung nomor baris item.
        row = index // columns

        # Menghitung nomor kolom item.
        column = index % columns

        # Menghitung posisi X item.
        x = 28 + column * column_width

        # Menghitung posisi Y item.
        y = item_top + row * 42

        # Mengambil warna wilayah.
        color = ADMIN_REGION_COLORS_BY_NAME.get(
            str(item["name"]),
            ADMIN_REGION_PALETTE[index % len(ADMIN_REGION_PALETTE)],
        )

        # Menggambar kotak warna wilayah.
        draw.rounded_rectangle((x, y + 4, x + 26, y + 30), radius=5, fill=(*color, 210))

        # Menggambar outline kotak warna.
        draw.rectangle((x, y + 4, x + 26, y + 30), outline=(0, 0, 0, 120), width=1)

        # Menggambar nama wilayah.
        draw.text((x + 36, y + 4), str(item["name"]), fill=(24, 49, 42, 255), font=item_font)

    # Mengembalikan gambar yang sudah memiliki legend.
    return output


# Fungsi utama untuk menyimpan overlay hasil prediksi di atas peta/citra referensi.
def save_overlay(
    # Probability map hasil model dengan nilai 0 sampai 1.
    probability: np.ndarray,
    # Citra referensi H0 jika peta dasar tidak tersedia.
    reference_input: Path,
    # Path tujuan file overlay.
    destination_path: Path,
    # Path peta dasar wilayah Riau jika tersedia.
    base_map_path: Path | None = None,
    # Opsi untuk menampilkan batas administrasi.
    include_admin_overlay: bool = True,
    # Opsi untuk menambahkan legend wilayah pada gambar.
    include_admin_legend: bool = False,
    # Opsi untuk menampilkan label nama wilayah.
    include_admin_labels: bool = False,
    # Opsi untuk menampilkan nomor wilayah.
    include_admin_numbers: bool = True,
) -> None:
    # Jika peta dasar tersedia, gunakan sebagai background overlay.
    if base_map_path and base_map_path.exists():
        # Membuka peta dasar.
        with Image.open(base_map_path) as base:
            # Mengubah peta dasar ke RGB.
            base_image = base.convert("RGB")

    # Jika peta dasar tidak tersedia, gunakan citra H0 sebagai background.
    else:
        # Membuka citra referensi H0.
        with Image.open(reference_input) as ref:
            # Mengubah citra referensi ke RGB.
            base_image = ref.convert("RGB")

    # Mengambil tinggi dan lebar probability map.
    h, w = probability.shape

    # Menyamakan ukuran background dengan probability map.
    base_image = base_image.resize((w, h), Image.BILINEAR)

    # Membuat overlay RGBA dari probability map.
    overlay_rgba = _to_overlay_rgba(
        probability=probability,
        min_visible_probability=settings.OVERLAY_MIN_VISIBLE_PROBABILITY,
    )

    # Mengubah array overlay RGBA menjadi object gambar.
    overlay_image = Image.fromarray(overlay_rgba)

    # Menggabungkan background dan overlay probabilitas.
    merged = Image.alpha_composite(base_image.convert("RGBA"), overlay_image)

    # Jika overlay administrasi aktif, gambar batas/label/nomor wilayah.
    if include_admin_overlay:
        # Menambahkan batas administrasi ke gambar.
        merged = _draw_admin_boundaries(
            merged,
            draw_labels=include_admin_labels,
            draw_numbers=include_admin_numbers,
        )

    # Jika legend diminta, tambahkan legend kabupaten/kota di bawah gambar.
    if include_admin_overlay and include_admin_legend:
        # Menambahkan area legend ke gambar.
        merged = _append_admin_legend(merged)

    # Menyimpan overlay final ke path tujuan.
    merged.save(destination_path)
