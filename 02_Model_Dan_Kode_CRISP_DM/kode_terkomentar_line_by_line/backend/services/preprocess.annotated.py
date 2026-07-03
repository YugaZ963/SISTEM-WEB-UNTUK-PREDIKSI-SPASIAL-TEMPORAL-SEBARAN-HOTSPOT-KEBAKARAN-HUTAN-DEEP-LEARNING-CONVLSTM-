# File anotasi dari `backend/services/preprocess.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Data Preparation, yaitu menyiapkan citra menjadi mask, sequence, dan patch.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Data preparation backend: validasi urutan input, ekstraksi hotspot red mask, dan pembentukan sequence H-6 sampai H0.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from dataclasses import dataclass
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image, ImageFilter


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@dataclass(frozen=True)
# Membuat wadah bernama `ModelSpec` untuk menyimpan data atau aturan kerja.
class ModelSpec:
    # Menjelaskan data `grid_size` yang disimpan atau dikirim pada bagian ini.
    grid_size: int
    # Menjelaskan data `channels` yang disimpan atau dikirim pada bagian ini.
    channels: int
    # Menjelaskan data `time_steps` yang disimpan atau dikirim pada bagian ini.
    time_steps: int
    # Menyimpan nilai ke `profile_name` untuk dipakai pada langkah berikutnya.
    profile_name: str = "legacy_128"
    # Menyimpan nilai ke `display_name` untuk dipakai pada langkah berikutnya.
    display_name: str = "Legacy ConvLSTM 128x128"
    # Menyimpan nilai ke `preprocess_mode` untuk dipakai pada langkah berikutnya.
    preprocess_mode: str = "grayscale_normalized"
    # Menyimpan nilai ke `inference_mode` untuk dipakai pada langkah berikutnya.
    inference_mode: str = "direct_resize"
    # Menyimpan nilai ke `patch_size` untuk dipakai pada langkah berikutnya.
    patch_size: int | None = None
    # Menyimpan nilai ke `patch_stride` untuk dipakai pada langkah berikutnya.
    patch_stride: int | None = None
    # Menyimpan nilai ke `patch_batch_size` untuk dipakai pada langkah berikutnya.
    patch_batch_size: int | None = None
    # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
    input_dilation_kernel: int = 3
    # Menyimpan nilai ke `recommended_threshold` untuk dipakai pada langkah berikutnya.
    recommended_threshold: float | None = None


# Membuat langkah kerja bernama `_normalize_kernel`.
def _normalize_kernel(size: int) -> int:
    # Menyimpan nilai ke `size` untuk dipakai pada langkah berikutnya.
    size = max(1, int(size))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return size if size % 2 == 1 else size + 1


# Membuat langkah kerja bernama `_extract_hotspot_red_mask`.
def _extract_hotspot_red_mask(image: Image.Image, dilation_kernel: int) -> np.ndarray:
    # Mengubah citra ke warna RGB agar format gambar seragam.
    rgb = image.convert("RGB")
    # Mengubah warna RGB ke HSV agar warna merah hotspot lebih mudah dipisahkan.
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)
    # Mengambil komponen warna H (hue) untuk mengenali warna merah.
    h = hsv[:, :, 0]
    # Mengambil komponen S (saturation) untuk memastikan warna cukup kuat.
    s = hsv[:, :, 1]
    # Mengambil komponen V (brightness) untuk memastikan piksel cukup terang.
    v = hsv[:, :, 2]

    # Catatan asli dari pembuat kode.
    # Threshold ini dibuat sama dengan script training historical risk patch.
    # Menandai piksel merah pada rentang warna merah bagian bawah.
    red_low = (h <= 14) & (s >= 70) & (v >= 50)
    # Menandai piksel merah pada rentang warna merah bagian atas.
    red_high = (h >= 242) & (s >= 70) & (v >= 50)
    # Menggabungkan hasil deteksi merah menjadi mask hotspot.
    mask = ((red_low | red_high).astype(np.uint8)) * 255

    # Mengubah mask angka menjadi gambar hitam-putih.
    mask_image = Image.fromarray(mask, mode="L")
    # Menyimpan nilai ke `dilation_kernel` untuk dipakai pada langkah berikutnya.
    dilation_kernel = _normalize_kernel(dilation_kernel)
    # Mengecek syarat sebelum melanjutkan proses.
    if dilation_kernel > 1:
        # Memperbesar titik hotspot kecil agar tidak mudah hilang.
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=dilation_kernel))

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.asarray(mask_image, dtype=np.float32) / 255.0


# Membuat langkah kerja bernama `_mask_to_channels`.
def _mask_to_channels(mask: np.ndarray, channels: int) -> np.ndarray:
    # Mengecek syarat sebelum melanjutkan proses.
    if channels == 1:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.expand_dims(mask, axis=-1)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return np.repeat(np.expand_dims(mask, axis=-1), repeats=channels, axis=-1)


# Membuat langkah kerja bernama `_load_image`.
def _load_image(
    # Menjelaskan data `path` yang disimpan atau dikirim pada bagian ini.
    path: Path,
    # Menjelaskan data `channels` yang disimpan atau dikirim pada bagian ini.
    channels: int,
    # Menjelaskan data `grid_size` yang disimpan atau dikirim pada bagian ini.
    grid_size: int,
    # Menjelaskan data `preprocess_mode` yang disimpan atau dikirim pada bagian ini.
    preprocess_mode: str,
    # Menjelaskan data `input_dilation_kernel` yang disimpan atau dikirim pada bagian ini.
    input_dilation_kernel: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> np.ndarray:
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with Image.open(path) as image:
        # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
        image = image.resize((grid_size, grid_size), Image.BILINEAR)

        # Mengecek syarat sebelum melanjutkan proses.
        if preprocess_mode == "hotspot_red_mask":
            # Menggabungkan hasil deteksi merah menjadi mask hotspot.
            mask = _extract_hotspot_red_mask(image=image, dilation_kernel=input_dilation_kernel)
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return _mask_to_channels(np.clip(mask, 0.0, 1.0), channels)

        # Mengecek syarat sebelum melanjutkan proses.
        if channels == 1:
            # Menyimpan nilai ke `gray` untuk dipakai pada langkah berikutnya.
            gray = image.convert("L")
            # Menyimpan nilai ke `array` untuk dipakai pada langkah berikutnya.
            array = np.asarray(gray, dtype=np.float32) / 255.0
            # Menyimpan nilai ke `array` untuk dipakai pada langkah berikutnya.
            array = np.expand_dims(array, axis=-1)
        # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
        else:
            # Mengubah citra ke warna RGB agar format gambar seragam.
            rgb = image.convert("RGB")
            # Menyimpan nilai ke `array` untuk dipakai pada langkah berikutnya.
            array = np.asarray(rgb, dtype=np.float32) / 255.0

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return array


# Membuat langkah kerja bernama `build_input_tensor`.
def build_input_tensor(
    # Menjelaskan data `ordered_input_paths` yang disimpan atau dikirim pada bagian ini.
    ordered_input_paths: dict[str, Path],
    # Menjelaskan data `required_stems` yang disimpan atau dikirim pada bagian ini.
    required_stems: list[str],
    # Menjelaskan data `spec` yang disimpan atau dikirim pada bagian ini.
    spec: ModelSpec,
    # Menyimpan nilai ke `preprocess_mode` untuk dipakai pada langkah berikutnya.
    preprocess_mode: str = "grayscale_normalized",
    # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
    input_dilation_kernel: int | None = None,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> np.ndarray:
    # Menyimpan nilai ke `sequence` untuk dipakai pada langkah berikutnya.
    sequence: list[np.ndarray] = []
    # Menyimpan nilai ke `dilation_kernel` untuk dipakai pada langkah berikutnya.
    dilation_kernel = input_dilation_kernel or spec.input_dilation_kernel
    # Mengulang proses untuk setiap data dalam daftar.
    for stem in required_stems:
        # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
        path = ordered_input_paths[stem]
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        sequence.append(
            # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
            _load_image(
                # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
                path=path,
                # Menyimpan nilai ke `channels` untuk dipakai pada langkah berikutnya.
                channels=spec.channels,
                # Menyimpan nilai ke `grid_size` untuk dipakai pada langkah berikutnya.
                grid_size=spec.grid_size,
                # Menyimpan nilai ke `preprocess_mode` untuk dipakai pada langkah berikutnya.
                preprocess_mode=preprocess_mode,
                # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
                input_dilation_kernel=dilation_kernel,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            )
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Menyimpan nilai ke `tensor` untuk dipakai pada langkah berikutnya.
    tensor = np.stack(sequence, axis=0).astype(np.float32)
    # Menyimpan nilai ke `tensor` untuk dipakai pada langkah berikutnya.
    tensor = np.expand_dims(tensor, axis=0)

    # Menyimpan nilai ke `expected_shape` untuk dipakai pada langkah berikutnya.
    expected_shape = (1, spec.time_steps, spec.grid_size, spec.grid_size, spec.channels)
    # Mengecek syarat sebelum melanjutkan proses.
    if tensor.shape != expected_shape:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError(f"Shape tensor tidak sesuai. Dapat {tensor.shape}, ekspektasi {expected_shape}.")

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tensor


# Membuat langkah kerja bernama `build_hotspot_mask_sequence`.
def build_hotspot_mask_sequence(
    # Menjelaskan data `ordered_input_paths` yang disimpan atau dikirim pada bagian ini.
    ordered_input_paths: dict[str, Path],
    # Menjelaskan data `required_stems` yang disimpan atau dikirim pada bagian ini.
    required_stems: list[str],
    # Menjelaskan data `input_dilation_kernel` yang disimpan atau dikirim pada bagian ini.
    input_dilation_kernel: int,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> np.ndarray:
    # Catatan pembuka/penutup yang menjelaskan isi bagian kode.
    """Build full-resolution mask sequence for patch-stitch inference."""
    # Menyimpan nilai ke `sequence` untuk dipakai pada langkah berikutnya.
    sequence: list[np.ndarray] = []
    # Menyimpan nilai ke `reference_size` untuk dipakai pada langkah berikutnya.
    reference_size: tuple[int, int] | None = None

    # Mengulang proses untuk setiap data dalam daftar.
    for stem in required_stems:
        # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
        path = ordered_input_paths[stem]
        # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
        with Image.open(path) as image:
            # Mengecek syarat sebelum melanjutkan proses.
            if reference_size is None:
                # Menyimpan nilai ke `reference_size` untuk dipakai pada langkah berikutnya.
                reference_size = image.size
            # Mengecek syarat sebelum melanjutkan proses.
            if image.size != reference_size:
                # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
                image = image.resize(reference_size, Image.BILINEAR)
            # Menggabungkan hasil deteksi merah menjadi mask hotspot.
            mask = _extract_hotspot_red_mask(image=image, dilation_kernel=input_dilation_kernel)
            # Melanjutkan langkah kerja pada bagian kode ini.
            sequence.append(np.clip(mask, 0.0, 1.0).astype(np.float32))

    # Menyimpan nilai ke `tensor` untuk dipakai pada langkah berikutnya.
    tensor = np.stack(sequence, axis=0).astype(np.float32)
    # Mengecek syarat sebelum melanjutkan proses.
    if tensor.ndim != 3 or tensor.shape[0] != len(required_stems):
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise ValueError(f"Shape sequence patch tidak sesuai: {tensor.shape}")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tensor
