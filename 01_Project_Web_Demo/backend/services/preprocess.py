"""Komentar file skripsi:
Data preparation backend untuk mengubah citra input menjadi mask hotspot merah sebelum masuk model ConvLSTM.

Konteks laporan: file ini mendukung BAB IV pada bagian data preparation,
karena mengubah 7 citra historis H-6 sampai H0 menjadi input numerik
yang siap diproses oleh model prediksi area risiko hotspot.
"""

# dataclass digunakan untuk membuat struktur data konfigurasi model secara ringkas dan jelas.
from dataclasses import dataclass

# Path digunakan untuk menerima dan membaca lokasi file citra input.
from pathlib import Path

# NumPy digunakan untuk mengubah citra menjadi array numerik yang dapat diproses model.
import numpy as np

# PIL Image digunakan untuk membuka, resize, konversi warna, dan memproses citra input.
from PIL import Image, ImageFilter


# Dataclass immutable untuk menyimpan spesifikasi model aktif yang dipakai saat preprocessing dan inferensi.
@dataclass(frozen=True)
class ModelSpec:
    # Ukuran grid input model, misalnya 128x128 untuk legacy atau 160x160 untuk patch model.
    grid_size: int

    # Jumlah channel citra input, misalnya 1 untuk grayscale/mask hotspot.
    channels: int

    # Jumlah frame waktu historis yang masuk ke model, yaitu 7 citra H-6 sampai H0.
    time_steps: int

    # Nama profile model aktif; default disiapkan untuk model lama.
    profile_name: str = "legacy_128"

    # Nama model yang ditampilkan pada dashboard atau status runtime.
    display_name: str = "Legacy ConvLSTM 128x128"

    # Mode preprocessing default; dapat diganti menjadi hotspot_red_mask pada model utama.
    preprocess_mode: str = "grayscale_normalized"

    # Mode inferensi default; direct_resize berarti citra langsung di-resize ke ukuran model.
    inference_mode: str = "direct_resize"

    # Ukuran patch jika model memakai mode patch_stitch.
    patch_size: int | None = None

    # Jarak pergeseran patch jika model memakai mode patch_stitch.
    patch_stride: int | None = None

    # Jumlah patch per batch saat inferensi.
    patch_batch_size: int | None = None

    # Ukuran kernel dilasi untuk memperluas area mask hotspot merah.
    input_dilation_kernel: int = 3

    # Threshold rekomendasi untuk mengubah probability map menjadi binary mask.
    recommended_threshold: float | None = None


# Fungsi internal untuk memastikan ukuran kernel selalu valid dan bernilai ganjil.
def _normalize_kernel(size: int) -> int:
    # Memastikan ukuran kernel minimal 1 dan bertipe integer.
    size = max(1, int(size))

    # Jika ukuran kernel sudah ganjil, nilai langsung digunakan; jika genap, ditambah 1.
    return size if size % 2 == 1 else size + 1


# Fungsi internal untuk mengekstraksi area hotspot merah dari citra menjadi mask numerik.
def _extract_hotspot_red_mask(image: Image.Image, dilation_kernel: int) -> np.ndarray:
    # Mengonversi citra ke RGB agar format warna awal seragam.
    rgb = image.convert("RGB")

    # Mengonversi RGB ke HSV karena warna merah lebih mudah dipisahkan berdasarkan hue, saturation, dan value.
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)

    # Channel H menyimpan informasi hue atau jenis warna.
    h = hsv[:, :, 0]

    # Channel S menyimpan tingkat kejenuhan warna.
    s = hsv[:, :, 1]

    # Channel V menyimpan tingkat kecerahan warna.
    v = hsv[:, :, 2]

    # Threshold ini dibuat sama dengan script training historical risk patch.
    # red_low mendeteksi warna merah pada rentang hue rendah.
    red_low = (h <= 14) & (s >= 70) & (v >= 50)

    # red_high mendeteksi warna merah pada rentang hue tinggi karena warna merah berada di dua ujung spektrum HSV.
    red_high = (h >= 242) & (s >= 70) & (v >= 50)

    # Menggabungkan dua rentang merah menjadi mask bernilai 255 untuk hotspot dan 0 untuk non-hotspot.
    mask = ((red_low | red_high).astype(np.uint8)) * 255

    # Mengubah array mask menjadi citra grayscale mode L agar bisa diproses filter PIL.
    mask_image = Image.fromarray(mask, mode="L")

    # Menormalisasi ukuran kernel agar sesuai syarat MaxFilter, yaitu bilangan ganjil.
    dilation_kernel = _normalize_kernel(dilation_kernel)

    # Jika kernel lebih dari 1, area hotspot diperlebar untuk memperjelas sinyal input model.
    if dilation_kernel > 1:
        # MaxFilter berfungsi sebagai proses dilasi sederhana pada mask hotspot.
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=dilation_kernel))

    # Mengubah mask menjadi array float32 dengan rentang 0 sampai 1.
    return np.asarray(mask_image, dtype=np.float32) / 255.0


# Fungsi internal untuk menyesuaikan mask 2D menjadi jumlah channel yang dibutuhkan model.
def _mask_to_channels(mask: np.ndarray, channels: int) -> np.ndarray:
    # Jika model memakai 1 channel, cukup menambahkan dimensi channel di akhir.
    if channels == 1:
        # Bentuk berubah dari (H, W) menjadi (H, W, 1).
        return np.expand_dims(mask, axis=-1)

    # Jika model memakai lebih dari 1 channel, mask diduplikasi ke setiap channel.
    return np.repeat(np.expand_dims(mask, axis=-1), repeats=channels, axis=-1)


# Fungsi internal untuk membuka satu citra dan mengubahnya menjadi array input model.
def _load_image(
    # Path file citra yang akan dibaca.
    path: Path,
    # Jumlah channel output yang dibutuhkan model.
    channels: int,
    # Ukuran grid model untuk mode direct_resize.
    grid_size: int,
    # Mode preprocessing, misalnya hotspot_red_mask atau grayscale_normalized.
    preprocess_mode: str,
    # Ukuran kernel dilasi untuk preprocessing hotspot_red_mask.
    input_dilation_kernel: int,
) -> np.ndarray:
    # Membuka file citra menggunakan PIL.
    with Image.open(path) as image:
        # Resize citra ke ukuran grid model agar shape input seragam.
        image = image.resize((grid_size, grid_size), Image.BILINEAR)

        # Jika mode preprocessing adalah hotspot_red_mask, sistem mengambil area merah hotspot.
        if preprocess_mode == "hotspot_red_mask":
            # Mengekstraksi mask merah hotspot dari citra.
            mask = _extract_hotspot_red_mask(image=image, dilation_kernel=input_dilation_kernel)

            # Mengubah mask menjadi jumlah channel yang sesuai dan membatasi nilai pada 0 sampai 1.
            return _mask_to_channels(np.clip(mask, 0.0, 1.0), channels)

        # Jika model membutuhkan 1 channel dan bukan mode hotspot_red_mask, citra diubah menjadi grayscale.
        if channels == 1:
            # Mengonversi citra menjadi grayscale.
            gray = image.convert("L")

            # Mengubah grayscale menjadi array float32 dengan rentang 0 sampai 1.
            array = np.asarray(gray, dtype=np.float32) / 255.0

            # Menambahkan dimensi channel sehingga shape menjadi (grid, grid, 1).
            array = np.expand_dims(array, axis=-1)

        # Jika model membutuhkan 3 channel, citra dipertahankan sebagai RGB.
        else:
            # Mengonversi citra menjadi RGB.
            rgb = image.convert("RGB")

            # Mengubah RGB menjadi array float32 dengan rentang 0 sampai 1.
            array = np.asarray(rgb, dtype=np.float32) / 255.0

    # Mengembalikan array citra yang sudah siap masuk ke sequence model.
    return array


# Fungsi untuk membentuk tensor input model pada mode direct_resize.
def build_input_tensor(
    # Dictionary path input yang sudah diurutkan berdasarkan stem H-6 sampai H0.
    ordered_input_paths: dict[str, Path],
    # Daftar stem wajib sebagai urutan waktu input model.
    required_stems: list[str],
    # Spesifikasi model aktif.
    spec: ModelSpec,
    # Mode preprocessing yang digunakan.
    preprocess_mode: str = "grayscale_normalized",
    # Kernel dilasi opsional; jika None memakai nilai dari ModelSpec.
    input_dilation_kernel: int | None = None,
) -> np.ndarray:
    # Menyiapkan list untuk menyimpan array setiap frame citra.
    sequence: list[np.ndarray] = []

    # Menentukan kernel dilasi yang digunakan saat preprocessing.
    dilation_kernel = input_dilation_kernel or spec.input_dilation_kernel

    # Membaca citra sesuai urutan waktu H-6, H-5, H-4, H-3, H-2, H-1, H0.
    for stem in required_stems:
        # Mengambil path file berdasarkan stem saat ini.
        path = ordered_input_paths[stem]

        # Memuat citra, melakukan resize, preprocessing, dan konversi ke array.
        sequence.append(
            _load_image(
                path=path,
                channels=spec.channels,
                grid_size=spec.grid_size,
                preprocess_mode=preprocess_mode,
                input_dilation_kernel=dilation_kernel,
            )
        )

    # Menumpuk 7 frame menjadi array sequence dengan shape (time, height, width, channels).
    tensor = np.stack(sequence, axis=0).astype(np.float32)

    # Menambahkan dimensi batch agar shape menjadi (1, time, height, width, channels).
    tensor = np.expand_dims(tensor, axis=0)

    # Menentukan shape tensor yang diharapkan oleh model ConvLSTM.
    expected_shape = (1, spec.time_steps, spec.grid_size, spec.grid_size, spec.channels)

    # Memastikan shape tensor sesuai spesifikasi model aktif.
    if tensor.shape != expected_shape:
        # Menghentikan proses jika shape input tidak sesuai kebutuhan model.
        raise ValueError(f"Shape tensor tidak sesuai. Dapat {tensor.shape}, ekspektasi {expected_shape}.")

    # Mengembalikan tensor input yang siap masuk ke model.
    return tensor


# Fungsi untuk membentuk sequence mask hotspot ukuran penuh pada mode patch_stitch.
def build_hotspot_mask_sequence(
    # Dictionary path input yang sudah diurutkan berdasarkan stem.
    ordered_input_paths: dict[str, Path],
    # Daftar stem wajib H-6 sampai H0.
    required_stems: list[str],
    # Ukuran kernel dilasi untuk memperjelas area hotspot merah.
    input_dilation_kernel: int,
) -> np.ndarray:
    """Build full-resolution mask sequence for patch-stitch inference."""

    # Menyiapkan list sequence untuk menyimpan mask hotspot setiap hari.
    sequence: list[np.ndarray] = []

    # Menyimpan ukuran referensi berdasarkan citra pertama agar seluruh frame konsisten.
    reference_size: tuple[int, int] | None = None

    # Membaca seluruh citra input berdasarkan urutan H-6 sampai H0.
    for stem in required_stems:
        # Mengambil path citra berdasarkan stem.
        path = ordered_input_paths[stem]

        # Membuka citra input.
        with Image.open(path) as image:
            # Jika citra pertama, ukurannya dijadikan ukuran referensi.
            if reference_size is None:
                reference_size = image.size

            # Jika ukuran citra berbeda dari referensi, citra disamakan ukurannya.
            if image.size != reference_size:
                # Resize dilakukan agar semua frame sequence memiliki dimensi spasial yang sama.
                image = image.resize(reference_size, Image.BILINEAR)

            # Mengekstraksi mask hotspot merah dari citra.
            mask = _extract_hotspot_red_mask(image=image, dilation_kernel=input_dilation_kernel)

            # Menyimpan mask dalam rentang 0 sampai 1 sebagai float32.
            sequence.append(np.clip(mask, 0.0, 1.0).astype(np.float32))

    # Menumpuk seluruh mask menjadi tensor 3D dengan shape (time, height, width).
    tensor = np.stack(sequence, axis=0).astype(np.float32)

    # Memastikan sequence patch berbentuk 3D dan jumlah frame sesuai jumlah input wajib.
    if tensor.ndim != 3 or tensor.shape[0] != len(required_stems):
        # Menghentikan proses jika shape sequence tidak sesuai.
        raise ValueError(f"Shape sequence patch tidak sesuai: {tensor.shape}")

    # Mengembalikan sequence mask hotspot untuk dipotong menjadi patch pada tahap inferensi.
    return tensor
