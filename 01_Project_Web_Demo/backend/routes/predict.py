"""Komentar file skripsi:
Endpoint API prediksi, riwayat, detail, unduh output, dan status runtime sistem web.

Konteks laporan: file ini mendukung BAB IV pada bagian implementasi sistem,
terutama proses deployment API, validasi input, inferensi model ConvLSTM,
penyimpanan hasil prediksi, visualisasi output, dan penyajian riwayat prediksi.
"""

# Mengimpor datetime untuk mencatat waktu prediksi, menghitung tanggal target H+1, dan memakai zona waktu UTC.
from datetime import datetime, timedelta, timezone

# Mengimpor Path untuk mengelola lokasi file/folder input, output, metadata, dan ZIP.
from pathlib import Path

# Mengimpor perf_counter untuk menghitung durasi proses inferensi dalam milidetik.
from time import perf_counter

# Mengimpor uuid4 untuk membuat prediction_id unik pada setiap proses prediksi.
from uuid import uuid4

# Mengimpor zipfile untuk membuat paket unduhan berisi input dan output prediksi.
import zipfile

# Mengimpor numpy untuk membaca probability map .npy dan memproses array hasil prediksi.
import numpy as np

# Mengimpor komponen FastAPI untuk router, upload file, form, request, dan error handling API.
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

# Mengimpor FileResponse untuk mengirim file ZIP hasil prediksi ke pengguna.
from fastapi.responses import FileResponse

# Mengimpor BaseModel dari Pydantic untuk mendefinisikan struktur response API.
from pydantic import BaseModel

# Mengimpor konfigurasi global backend seperti folder storage, batas file, dan profile model aktif.
from backend import settings

# Mengimpor fungsi postprocessing untuk membuat heatmap, overlay, binary mask, dan legend wilayah.
from backend.services.postprocess import (
    # Mengambil daftar legenda kabupaten/kota Riau untuk tampilan dashboard.
    get_admin_region_legend,
    # Menyimpan binary mask berdasarkan threshold probabilitas.
    save_binary_mask,
    # Menyimpan heatmap dari probability map hasil prediksi.
    save_heatmap,
    # Menyimpan overlay prediksi di atas peta dasar atau citra referensi H0.
    save_overlay,
    # Menyimpan probability map dalam format .npy.
    save_probability_array,
)

# Mengimpor fungsi preprocessing untuk membentuk input model dari 7 citra historis.
from backend.services.preprocess import build_hotspot_mask_sequence, build_input_tensor

# Mengimpor fungsi storage untuk membuat folder, membaca/menulis JSON, menyimpan upload, dan membuat URL publik.
from backend.services.storage import (
    # Membuat folder input dan output berdasarkan prediction_id.
    create_prediction_dirs,
    # Membaca metadata JSON hasil prediksi.
    read_json,
    # Menyimpan file upload yang sudah divalidasi.
    save_upload_files,
    # Mengubah path file storage menjadi URL publik /storage.
    to_public_storage_url,
    # Menulis metadata prediksi ke file JSON.
    write_json,
)

# Mengimpor validator untuk memastikan 7 file input sesuai format H-6 sampai H0.
from backend.services.validator import validate_and_order_files

# Membuat router FastAPI dengan prefix /api agar seluruh endpoint berada pada namespace API.
router = APIRouter(prefix="/api", tags=["prediction"])

# Versi gaya visual digunakan untuk menandai apakah overlay lama perlu diperbarui.
VISUAL_STYLE_VERSION = "admin-number-toggle-v4-20260617"


# Model response untuk setiap output prediksi, misalnya H+1.
class PredictionOutput(BaseModel):
    # Nomor langkah prediksi, contoh 1 untuk H+1.
    step: int

    # URL heatmap hasil prediksi.
    heatmap_url: str

    # URL overlay dengan batas/nomor wilayah administrasi.
    overlay_url: str | None = None

    # URL overlay batas wilayah tanpa nomor wilayah.
    overlay_no_numbers_url: str | None = None

    # URL overlay bersih tanpa batas administrasi.
    overlay_plain_url: str | None = None

    # URL binary mask jika user mengisi threshold.
    binary_url: str | None = None


# Model response utama endpoint POST /api/predict.
class PredictResponse(BaseModel):
    # ID unik untuk satu proses prediksi.
    prediction_id: str

    # URL heatmap utama, biasanya output H+1.
    heatmap_url: str

    # URL overlay utama dengan batas/nomor wilayah.
    overlay_url: str | None = None

    # URL overlay tanpa nomor wilayah.
    overlay_no_numbers_url: str | None = None

    # URL overlay bersih tanpa batas wilayah.
    overlay_plain_url: str | None = None

    # URL untuk mengunduh seluruh input dan output dalam format ZIP.
    download_zip_url: str | None = None

    # Durasi proses prediksi dalam milidetik.
    processing_time_ms: int

    # Ukuran grid/model aktif.
    grid_size: int

    # Threshold yang dipakai user untuk binary mask; None jika tidak diisi.
    threshold: float | None = None

    # Horizon prediksi; sistem ini dibatasi untuk H+1.
    horizon: int

    # Nama engine prediksi yang aktif, ConvLSTM atau fallback heuristik.
    prediction_engine: str

    # Nama backend model untuk kompatibilitas response frontend.
    model_backend: str | None = None

    # Mode preprocessing yang digunakan, misalnya hotspot_red_mask.
    preprocess_mode: str | None = None

    # Profile model aktif, misalnya historical_risk_patch_160.
    model_profile: str | None = None

    # Mode inferensi, misalnya patch_stitch atau direct_resize.
    inference_mode: str | None = None

    # Path model yang berhasil dimuat, jika tersedia.
    model_path: str | None = None

    # Threshold rekomendasi berdasarkan konfigurasi model.
    recommended_threshold: float | None = None

    # Bentuk output probability map hasil prediksi.
    output_shape: list[int] | None = None

    # Tanggal input dari nama file jika format bertanggal digunakan.
    input_dates: dict[str, str] | None = None

    # Tanggal target prediksi H+1 jika tanggal H0 tersedia.
    target_date: str | None = None

    # Daftar legenda wilayah administrasi untuk dashboard.
    admin_regions: list[dict] | None = None

    # Daftar seluruh output prediksi.
    outputs: list[PredictionOutput]


# Helper untuk membaca metadata prediksi atau mengembalikan error 404 jika ID tidak ditemukan.
def _load_prediction_or_404(prediction_id: str) -> tuple[dict, Path, Path]:
    # Menentukan folder input berdasarkan prediction_id.
    input_dir = settings.INPUTS_DIR / prediction_id

    # Menentukan folder output berdasarkan prediction_id.
    output_dir = settings.OUTPUTS_DIR / prediction_id

    # Menentukan lokasi file metadata params.json.
    params_path = output_dir / "params.json"

    # Memeriksa apakah metadata prediksi tersedia.
    if not params_path.exists():
        # Mengembalikan error 404 jika prediction_id tidak ditemukan.
        raise HTTPException(
            # Status 404 menunjukkan data prediksi tidak ada.
            status_code=status.HTTP_404_NOT_FOUND,
            # Pesan error dikirim agar frontend/user mengetahui masalahnya.
            detail=f"Prediction ID '{prediction_id}' tidak ditemukan.",
        )

    # Mengembalikan metadata, folder input, dan folder output.
    return read_json(params_path), input_dir, output_dir


# Helper untuk membuat ZIP berisi file input dan output dari satu prediksi.
def _build_prediction_zip(zip_path: Path, input_dir: Path, output_dir: Path) -> None:
    # Membuka file ZIP dalam mode tulis dengan kompresi.
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Membaca seluruh file pada folder input.
        for path in sorted(input_dir.glob("*")):
            # Hanya file yang dimasukkan ke ZIP.
            if path.is_file():
                # Menambahkan input ke folder virtual inputs/ dalam ZIP.
                archive.write(path, arcname=f"inputs/{path.name}")

        # Membaca seluruh file pada folder output.
        for path in sorted(output_dir.glob("*")):
            # Melewati item yang bukan file.
            if not path.is_file():
                continue

            # Melewati file ZIP yang sedang dibuat agar tidak masuk ke dirinya sendiri.
            if path == zip_path:
                continue

            # Menambahkan output ke folder virtual outputs/ dalam ZIP.
            archive.write(path, arcname=f"outputs/{path.name}")


# Helper untuk mencari path file H0 sebagai referensi overlay.
def _reference_h0_path(data: dict, input_dir: Path) -> Path | None:
    # Mengambil daftar file input dari metadata.
    input_files = data.get("input_files", {})

    # Memastikan input_files berbentuk dictionary.
    if isinstance(input_files, dict):
        # Mengambil nama file untuk stem H0.
        file_name = input_files.get("H0")

        # Memastikan nama file H0 valid sebagai string.
        if isinstance(file_name, str):
            # Membentuk kandidat path H0 dari folder input.
            candidate = input_dir / file_name

            # Mengembalikan path jika file H0 ditemukan.
            if candidate.exists():
                return candidate

    # Fallback untuk format nama lama seperti H0.png, H0.jpg, atau H0.jpeg.
    for suffix in (".png", ".jpg", ".jpeg"):
        # Membentuk kandidat path berdasarkan ekstensi.
        candidate = input_dir / f"H0{suffix}"

        # Mengembalikan kandidat jika file tersedia.
        if candidate.exists():
            return candidate

    # Mengembalikan None jika file referensi H0 tidak ditemukan.
    return None


# Helper untuk memastikan visual output lama tetap lengkap dan sesuai versi visual terbaru.
def _ensure_prediction_visuals(data: dict, input_dir: Path, output_dir: Path) -> dict:
    # Menyalin metadata agar tidak mengubah object asli secara langsung.
    payload = dict(data)

    # Mengambil daftar output dari metadata.
    outputs = payload.get("outputs", [])

    # Jika outputs bukan list, sistem memakai list kosong.
    if not isinstance(outputs, list):
        outputs = []

    # Penanda apakah metadata berubah dan perlu ditulis ulang.
    changed = False

    # Threshold efektif untuk membuat binary mask saat membuka riwayat/detail.
    effective_threshold = payload.get("threshold")
    if effective_threshold is None:
        effective_threshold = payload.get("recommended_threshold", 0.55)
        payload["threshold"] = effective_threshold
        changed = True

    # Mencari file H0 sebagai background/referensi overlay.
    reference_h0 = _reference_h0_path(payload, input_dir)

    # Menyiapkan list output yang sudah dinormalisasi.
    normalized_outputs: list[dict] = []

    # Memproses setiap output prediksi, misalnya H+1.
    for index, output in enumerate(outputs, start=1):
        # Melewati output yang tidak berbentuk dictionary.
        if not isinstance(output, dict):
            continue

        # Menyalin item output agar dapat diperbaiki tanpa merusak data asli.
        item = dict(output)

        # Mengambil nomor step prediksi; jika tidak ada memakai urutan index.
        step = item.get("step") or index

        # Membentuk nama step seperti H+1.
        step_name = f"H+{step}"

        # Mengambil nama file probability map atau membuat nama default.
        probability_file = item.get("probability_file") or f"probability_{step_name}.npy"

        # Menentukan path probability map.
        probability_path = output_dir / str(probability_file)

        # Mengambil nama file heatmap atau membuat nama default.
        heatmap_file = item.get("heatmap_file") or f"heatmap_{step_name}.png"

        # Menentukan path heatmap.
        heatmap_path = output_dir / str(heatmap_file)

        # Jika heatmap tersedia tetapi URL belum ada, URL ditambahkan ke metadata.
        if heatmap_path.exists() and not item.get("heatmap_url"):
            # Membuat URL publik heatmap.
            item["heatmap_url"] = to_public_storage_url(heatmap_path)
            # Menandai metadata berubah.
            changed = True

        # Mengambil nama file overlay utama atau membuat nama default.
        overlay_file = item.get("overlay_file") or f"overlay_{step_name}.png"

        # Menentukan path overlay utama.
        overlay_path = output_dir / str(overlay_file)

        # Mengecek apakah visual lama perlu dibuat ulang karena versi style berubah.
        needs_overlay_refresh = payload.get("visual_style_version") != VISUAL_STYLE_VERSION

        # Jika perlu refresh dan probability tersedia, overlay utama dibuat ulang.
        if needs_overlay_refresh and probability_path.exists() and reference_h0 is not None:
            # Membaca probability map dari file .npy.
            probability = np.load(probability_path)

            # Menyimpan overlay dengan batas administrasi dan nomor wilayah.
            save_overlay(
                probability=probability,
                reference_input=reference_h0,
                destination_path=overlay_path,
                base_map_path=settings.BASE_MAP_PATH,
                include_admin_overlay=True,
                include_admin_legend=False,
                include_admin_labels=False,
                include_admin_numbers=True,
            )

            # Menandai metadata berubah.
            changed = True

        # Jika overlay tersedia tetapi URL belum ada, URL ditambahkan.
        if overlay_path.exists() and not item.get("overlay_url"):
            # Membuat URL publik overlay.
            item["overlay_url"] = to_public_storage_url(overlay_path)
            # Menandai metadata berubah.
            changed = True

        # Mengambil nama file overlay tanpa nomor atau membuat nama default.
        overlay_no_numbers_file = item.get("overlay_no_numbers_file") or f"overlay_no_numbers_{step_name}.png"

        # Menentukan path overlay tanpa nomor.
        overlay_no_numbers_path = output_dir / str(overlay_no_numbers_file)

        # Jika overlay tanpa nomor belum ada atau style berubah, file dibuat ulang.
        if (needs_overlay_refresh or not overlay_no_numbers_path.exists()) and probability_path.exists() and reference_h0 is not None:
            # Membaca probability map.
            probability = np.load(probability_path)

            # Menyimpan overlay dengan batas administrasi tetapi tanpa nomor wilayah.
            save_overlay(
                probability=probability,
                reference_input=reference_h0,
                destination_path=overlay_no_numbers_path,
                base_map_path=settings.BASE_MAP_PATH,
                include_admin_overlay=True,
                include_admin_legend=False,
                include_admin_labels=False,
                include_admin_numbers=False,
            )

            # Menandai metadata berubah.
            changed = True

        # Jika overlay tanpa nomor tersedia, metadata file dan URL disinkronkan.
        if overlay_no_numbers_path.exists():
            # Memperbarui nama file jika belum sama.
            if item.get("overlay_no_numbers_file") != overlay_no_numbers_path.name:
                item["overlay_no_numbers_file"] = overlay_no_numbers_path.name
                changed = True

            # Memperbarui URL jika belum sama.
            if item.get("overlay_no_numbers_url") != to_public_storage_url(overlay_no_numbers_path):
                item["overlay_no_numbers_url"] = to_public_storage_url(overlay_no_numbers_path)
                changed = True

        # Mengambil nama file overlay bersih atau membuat nama default.
        overlay_plain_file = item.get("overlay_plain_file") or f"overlay_plain_{step_name}.png"

        # Menentukan path overlay bersih.
        overlay_plain_path = output_dir / str(overlay_plain_file)

        # Jika overlay bersih belum ada, file dibuat dari probability map dan H0.
        if not overlay_plain_path.exists() and probability_path.exists() and reference_h0 is not None:
            # Membaca probability map.
            probability = np.load(probability_path)

            # Menyimpan overlay bersih tanpa batas administrasi.
            save_overlay(
                probability=probability,
                reference_input=reference_h0,
                destination_path=overlay_plain_path,
                base_map_path=settings.BASE_MAP_PATH,
                include_admin_overlay=False,
                include_admin_legend=False,
                include_admin_labels=False,
            )

            # Menandai metadata berubah.
            changed = True

        # Jika overlay bersih tersedia, metadata file dan URL disinkronkan.
        if overlay_plain_path.exists():
            # Memperbarui nama file overlay bersih.
            if item.get("overlay_plain_file") != overlay_plain_path.name:
                item["overlay_plain_file"] = overlay_plain_path.name
                changed = True

            # Memperbarui URL overlay bersih.
            if item.get("overlay_plain_url") != to_public_storage_url(overlay_plain_path):
                item["overlay_plain_url"] = to_public_storage_url(overlay_plain_path)
                changed = True

        # Mengambil nama file binary mask atau membuat nama default.
        binary_file = item.get("binary_file") or f"binary_{step_name}.png"

        # Menentukan path binary mask.
        binary_path = output_dir / str(binary_file)

        # Jika binary belum ada tetapi probability tersedia, buat dari threshold efektif.
        if effective_threshold is not None and not binary_path.exists() and probability_path.exists():
            # Membaca probability map.
            probability = np.load(probability_path)

            # Menyimpan binary mask berdasarkan threshold efektif.
            save_binary_mask(
                probability=probability,
                threshold=float(effective_threshold),
                path=binary_path,
            )

            # Menandai metadata berubah.
            changed = True

        # Jika binary mask tersedia, metadata file dan URL disinkronkan.
        if binary_path.exists():
            # Memperbarui nama file binary mask.
            if item.get("binary_file") != binary_path.name:
                item["binary_file"] = binary_path.name
                changed = True

            # Memperbarui URL binary mask.
            if item.get("binary_url") != to_public_storage_url(binary_path):
                item["binary_url"] = to_public_storage_url(binary_path)
                changed = True

        # Menambahkan item yang sudah dinormalisasi ke daftar output.
        normalized_outputs.append(item)

    # Jika output berhasil dinormalisasi, metadata output utama ikut diperbarui.
    if normalized_outputs:
        # Menyimpan daftar output terbaru.
        payload["outputs"] = normalized_outputs

        # Mengambil output pertama sebagai output utama dashboard.
        first = normalized_outputs[0]

        # Mengisi URL heatmap utama jika belum ada.
        payload.setdefault("heatmap_url", first.get("heatmap_url"))

        # Mengisi URL overlay utama jika belum ada.
        payload.setdefault("overlay_url", first.get("overlay_url"))

        # Mengisi URL overlay tanpa nomor jika belum ada.
        payload.setdefault("overlay_no_numbers_url", first.get("overlay_no_numbers_url"))

        # Mengisi URL overlay bersih jika belum ada.
        payload.setdefault("overlay_plain_url", first.get("overlay_plain_url"))

        # Mengisi URL binary mask utama jika belum ada.
        payload.setdefault("binary_url", first.get("binary_url"))

    # Mengambil legend kabupaten/kota untuk metadata.
    admin_regions = get_admin_region_legend()

    # Memperbarui legend jika berbeda dari metadata lama.
    if payload.get("admin_regions") != admin_regions:
        payload["admin_regions"] = admin_regions
        changed = True

    # Memperbarui versi visual jika belum sesuai.
    if payload.get("visual_style_version") != VISUAL_STYLE_VERSION:
        payload["visual_style_version"] = VISUAL_STYLE_VERSION
        changed = True

    # Jika metadata berubah, tulis ulang params.json di output dan input.
    if changed:
        # Menulis metadata terbaru ke folder output.
        write_json(output_dir / "params.json", payload)

        # Jika folder input ada, salin metadata yang sama ke folder input.
        if input_dir.exists():
            write_json(input_dir / "params.json", payload)

    # Mengembalikan metadata yang sudah dipastikan lengkap.
    return payload


# Helper untuk memperkaya detail prediksi sebelum dikirim ke frontend.
def _enrich_prediction_detail(data: dict, prediction_id: str, input_dir: Path, output_dir: Path) -> dict:
    # Memastikan output visual dan metadata sudah lengkap.
    data = _ensure_prediction_visuals(data=data, input_dir=input_dir, output_dir=output_dir)

    # Menyalin data agar aman untuk ditambah informasi detail.
    payload = dict(data)

    # Mengambil daftar output prediksi.
    outputs = payload.get("outputs", [])

    # Jika output tersedia, output pertama dijadikan output utama detail.
    if isinstance(outputs, list) and outputs:
        # Mengambil output pertama.
        first = outputs[0]

        # Memastikan output pertama berbentuk dictionary.
        if isinstance(first, dict):
            # Mengisi URL heatmap utama jika belum ada.
            payload.setdefault("heatmap_url", first.get("heatmap_url"))

            # Mengisi URL overlay utama jika belum ada.
            payload.setdefault("overlay_url", first.get("overlay_url"))

            # Mengisi URL overlay tanpa nomor jika belum ada.
            payload.setdefault("overlay_no_numbers_url", first.get("overlay_no_numbers_url"))

            # Mengisi URL overlay bersih jika belum ada.
            payload.setdefault("overlay_plain_url", first.get("overlay_plain_url"))

    # Memastikan legend wilayah tersedia pada detail.
    payload.setdefault("admin_regions", get_admin_region_legend())

    # Mengambil daftar nama file input dari metadata.
    input_files = payload.get("input_files", {})

    # Menyiapkan dictionary URL input untuk ditampilkan di frontend.
    input_urls: dict[str, str] = {}

    # Jika input_files valid, buat URL publik untuk setiap file input.
    if isinstance(input_files, dict):
        # Melakukan iterasi setiap stem dan nama file input.
        for stem, file_name in input_files.items():
            # Melewati data nama file yang bukan string.
            if not isinstance(file_name, str):
                continue

            # Membentuk path input berdasarkan nama file.
            file_path = input_dir / file_name

            # Jika file input ada, buat URL publik.
            if file_path.exists():
                input_urls[stem] = to_public_storage_url(file_path)

    # Menambahkan URL input ke payload detail.
    payload["input_urls"] = input_urls

    # Menambahkan URL detail prediksi.
    payload["detail_url"] = f"/api/predictions/{prediction_id}"

    # Menambahkan URL download ZIP prediksi.
    payload["download_zip_url"] = f"/api/predictions/{prediction_id}/download"

    # Mengembalikan payload detail yang sudah lengkap.
    return payload


# Endpoint utama untuk menerima 7 citra, menjalankan prediksi, dan mengembalikan hasil.
@router.post("/predict", response_model=PredictResponse)
async def predict(
    # Request dipakai untuk mengakses app.state, yaitu model_spec dan inference_service.
    request: Request,
    # files menerima banyak file upload dari form-data.
    files: list[UploadFile] = File(...),
    # threshold bersifat opsional untuk membuat binary mask.
    threshold: float | None = Form(default=None),
    # horizon default 1 karena sistem saat ini hanya mendukung H+1.
    horizon: int = Form(default=1),
) -> PredictResponse:
    # Memvalidasi threshold agar berada pada rentang probabilitas 0 sampai 1.
    if threshold is not None and not (0.0 <= threshold <= 1.0):
        # Mengembalikan error jika threshold tidak valid.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nilai threshold harus pada rentang 0 sampai 1.",
        )

    # Membatasi horizon prediksi hanya H+1 sesuai batasan sistem web.
    if horizon != 1:
        # Mengembalikan error jika user meminta horizon selain 1.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sistem saat ini hanya mendukung prediksi H+1.",
        )

    # Memvalidasi jumlah, nama, ekstensi, ukuran, MIME type, dan urutan file H-6 sampai H0.
    ordered_files = await validate_and_order_files(
        files=files,
        required_stems=settings.REQUIRED_STEMS,
        allowed_extensions=settings.ALLOWED_EXTENSIONS,
        max_file_bytes=settings.MAX_FILE_BYTES,
    )

    # Mengambil tanggal input dari nama file jika user memakai format H-6_YYYY-MM-DD.png.
    input_dates = {
        record.stem: record.acquired_date.isoformat()
        for record in ordered_files
        if record.acquired_date is not None
    }

    # Menyiapkan tanggal target prediksi H+1.
    target_date = None

    # Jika tanggal input tersedia, tanggal H+1 dihitung dari tanggal H0 + 1 hari.
    if input_dates:
        # Mengambil tanggal H0 dari file terakhir.
        h0_date = ordered_files[-1].acquired_date

        # Jika tanggal H0 valid, hitung target_date.
        if h0_date is not None:
            target_date = (h0_date + timedelta(days=1)).isoformat()

    # Membuat ID unik untuk proses prediksi.
    prediction_id = str(uuid4())

    # Membuat folder input dan output berdasarkan prediction_id.
    input_dir, output_dir = create_prediction_dirs(prediction_id)

    # Menyimpan file upload ke folder input.
    saved_inputs = await save_upload_files(ordered_files=ordered_files, input_dir=input_dir)

    # Membuat metadata nama file input.
    metadata_inputs = {stem: path.name for stem, path in saved_inputs.items()}

    # Mengambil spesifikasi model aktif dari app.state.
    model_spec = request.app.state.model_spec

    # Jika threshold kosong, gunakan threshold rekomendasi model agar binary mask selalu tersedia.
    if threshold is None:
        threshold = model_spec.recommended_threshold if model_spec.recommended_threshold is not None else 0.55

    # Mengambil service inferensi yang sudah dimuat saat aplikasi start.
    inference_service = request.app.state.inference_service

    # Mulai menghitung waktu proses inferensi.
    start = perf_counter()

    # Jika model memakai patch_stitch, input dibuat sebagai sequence mask ukuran asli.
    if model_spec.inference_mode == "patch_stitch":
        # Membentuk sequence mask hotspot merah dari 7 citra.
        sequence = build_hotspot_mask_sequence(
            ordered_input_paths=saved_inputs,
            required_stems=settings.REQUIRED_STEMS,
            input_dilation_kernel=model_spec.input_dilation_kernel,
        )

        # Menjalankan prediksi horizon menggunakan service inferensi.
        predicted_maps = inference_service.predict_horizon(sequence=sequence, horizon=horizon)

    # Jika bukan patch_stitch, input dibuat sebagai tensor resize langsung.
    else:
        # Membentuk tensor input model dengan shape sesuai ModelSpec.
        tensor = build_input_tensor(
            ordered_input_paths=saved_inputs,
            required_stems=settings.REQUIRED_STEMS,
            spec=model_spec,
            preprocess_mode=inference_service.preprocess_mode,
            input_dilation_kernel=model_spec.input_dilation_kernel,
        )

        # Menjalankan prediksi horizon menggunakan tensor.
        predicted_maps = inference_service.predict_horizon(sequence=tensor, horizon=horizon)

    # Menghitung durasi proses inferensi dalam milidetik.
    duration_ms = int((perf_counter() - start) * 1000)

    # Menggunakan citra H0 sebagai referensi tampilan overlay.
    reference_h0 = saved_inputs["H0"]

    # Menyiapkan daftar response output.
    outputs: list[PredictionOutput] = []

    # Menyiapkan daftar metadata output untuk disimpan ke params.json.
    outputs_metadata: list[dict] = []

    # Memproses setiap probability map hasil prediksi.
    for index, probability in enumerate(predicted_maps, start=1):
        # Membentuk nama step prediksi, misalnya H+1.
        step_name = f"H+{index}"

        # Menentukan path file probability map.
        prob_path = output_dir / f"probability_{step_name}.npy"

        # Menentukan path file heatmap.
        heatmap_path = output_dir / f"heatmap_{step_name}.png"

        # Menentukan path file overlay utama.
        overlay_path = output_dir / f"overlay_{step_name}.png"

        # Menentukan path file overlay tanpa nomor wilayah.
        overlay_no_numbers_path = output_dir / f"overlay_no_numbers_{step_name}.png"

        # Menentukan path file overlay bersih.
        overlay_plain_path = output_dir / f"overlay_plain_{step_name}.png"

        # Menyimpan probability map dalam format .npy.
        save_probability_array(probability=probability, path=prob_path)

        # Menyimpan heatmap prediksi.
        save_heatmap(probability=probability, path=heatmap_path)

        # Menyimpan overlay utama dengan batas administrasi dan nomor wilayah.
        save_overlay(
            probability=probability,
            reference_input=reference_h0,
            destination_path=overlay_path,
            base_map_path=settings.BASE_MAP_PATH,
            include_admin_overlay=True,
            include_admin_legend=False,
            include_admin_labels=False,
            include_admin_numbers=True,
        )

        # Menyimpan overlay dengan batas administrasi tetapi tanpa nomor wilayah.
        save_overlay(
            probability=probability,
            reference_input=reference_h0,
            destination_path=overlay_no_numbers_path,
            base_map_path=settings.BASE_MAP_PATH,
            include_admin_overlay=True,
            include_admin_legend=False,
            include_admin_labels=False,
            include_admin_numbers=False,
        )

        # Menyimpan overlay bersih tanpa batas administrasi.
        save_overlay(
            probability=probability,
            reference_input=reference_h0,
            destination_path=overlay_plain_path,
            base_map_path=settings.BASE_MAP_PATH,
            include_admin_overlay=False,
            include_admin_legend=False,
            include_admin_labels=False,
        )

        # Menyiapkan URL binary mask; default None jika threshold tidak diisi.
        binary_url = None

        # Menyiapkan path binary mask; default None jika threshold tidak diisi.
        binary_path = None

        # Jika threshold diisi user, sistem membuat binary mask.
        if threshold is not None:
            # Menentukan path binary mask.
            binary_path = output_dir / f"binary_{step_name}.png"

            # Menyimpan binary mask berdasarkan threshold.
            save_binary_mask(probability=probability, threshold=threshold, path=binary_path)

            # Membuat URL publik binary mask.
            binary_url = to_public_storage_url(binary_path)

        # Membentuk object response untuk satu output prediksi.
        output_entry = PredictionOutput(
            step=index,
            heatmap_url=to_public_storage_url(heatmap_path),
            overlay_url=to_public_storage_url(overlay_path),
            overlay_no_numbers_url=to_public_storage_url(overlay_no_numbers_path),
            overlay_plain_url=to_public_storage_url(overlay_plain_path),
            binary_url=binary_url,
        )

        # Menambahkan output ke daftar response.
        outputs.append(output_entry)

        # Menambahkan metadata output untuk disimpan ke params.json.
        outputs_metadata.append(
            {
                "step": index,
                "heatmap_url": output_entry.heatmap_url,
                "overlay_url": output_entry.overlay_url,
                "overlay_no_numbers_url": output_entry.overlay_no_numbers_url,
                "overlay_plain_url": output_entry.overlay_plain_url,
                "binary_url": output_entry.binary_url,
                "probability_file": prob_path.name,
                "heatmap_file": heatmap_path.name,
                "overlay_file": overlay_path.name,
                "overlay_no_numbers_file": overlay_no_numbers_path.name,
                "overlay_plain_file": overlay_plain_path.name,
                "binary_file": binary_path.name if binary_path else None,
            }
        )

    # Mengambil legenda wilayah administrasi Riau.
    admin_regions = get_admin_region_legend()

    # Menyusun metadata lengkap proses prediksi.
    metadata = {
        "prediction_id": prediction_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "recommended_threshold": model_spec.recommended_threshold,
        "horizon": horizon,
        "grid_size": model_spec.grid_size,
        "channels": model_spec.channels,
        "model_profile": model_spec.profile_name,
        "model_display_name": model_spec.display_name,
        "inference_mode": model_spec.inference_mode,
        "patch_size": model_spec.patch_size,
        "patch_stride": model_spec.patch_stride,
        "input_dilation_kernel": model_spec.input_dilation_kernel,
        "prediction_engine": inference_service.backend,
        "model_backend": inference_service.backend,
        "model_path": inference_service.model_path,
        "processing_time_ms": duration_ms,
        "preprocess_mode": inference_service.preprocess_mode,
        "output_shape": list(predicted_maps[0].shape) if predicted_maps else None,
        "input_files": metadata_inputs,
        "input_dates": input_dates or None,
        "target_date": target_date,
        "admin_regions": admin_regions,
        "visual_style_version": VISUAL_STYLE_VERSION,
        "outputs": outputs_metadata,
    }

    # Menyimpan metadata pada folder input agar input dan parameter prediksi terdokumentasi.
    write_json(input_dir / "params.json", metadata)

    # Menyimpan metadata pada folder output sebagai sumber riwayat/detail prediksi.
    write_json(output_dir / "params.json", metadata)

    # Mengambil output pertama sebagai output utama response.
    first = outputs[0]

    # Mengembalikan response prediksi ke frontend.
    return PredictResponse(
        prediction_id=prediction_id,
        heatmap_url=first.heatmap_url,
        overlay_url=first.overlay_url,
        overlay_no_numbers_url=first.overlay_no_numbers_url,
        overlay_plain_url=first.overlay_plain_url,
        download_zip_url=f"/api/predictions/{prediction_id}/download",
        processing_time_ms=duration_ms,
        grid_size=model_spec.grid_size,
        threshold=threshold,
        horizon=horizon,
        prediction_engine=inference_service.backend,
        model_backend=inference_service.backend,
        preprocess_mode=inference_service.preprocess_mode,
        model_profile=model_spec.profile_name,
        inference_mode=model_spec.inference_mode,
        model_path=inference_service.model_path,
        recommended_threshold=model_spec.recommended_threshold,
        output_shape=list(predicted_maps[0].shape) if predicted_maps else None,
        input_dates=input_dates or None,
        target_date=target_date,
        admin_regions=admin_regions,
        outputs=outputs,
    )


# Endpoint untuk mengambil daftar riwayat prediksi terbaru.
@router.get("/predictions")
def list_predictions(limit: int = 20) -> list[dict]:
    # Memastikan limit minimal 1 agar query riwayat tetap valid.
    if limit < 1:
        # Mengembalikan error jika limit tidak valid.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter limit minimal 1.",
        )

    # Mengambil seluruh file params.json dari folder output dan mengurutkan dari yang terbaru.
    params_files = sorted(
        settings.OUTPUTS_DIR.glob("*/params.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    # Menyiapkan list record riwayat.
    records: list[dict] = []

    # Membaca setiap file metadata prediksi.
    for params_file in params_files:
        # Membaca metadata dari params.json.
        data = read_json(params_file)

        # Melewati riwayat yang bukan H+1.
        if data.get("horizon", 1) != 1:
            continue

        # Mengambil prediction_id dari metadata atau nama folder.
        prediction_id = data.get("prediction_id") or params_file.parent.name

        # Menentukan folder input prediksi.
        input_dir = settings.INPUTS_DIR / str(prediction_id)

        # Menentukan folder output prediksi.
        output_dir = params_file.parent

        # Memastikan metadata visual tetap lengkap dan terbaru.
        data = _ensure_prediction_visuals(data=data, input_dir=input_dir, output_dir=output_dir)

        # Menambahkan ringkasan prediksi ke daftar riwayat.
        records.append(
            {
                "prediction_id": prediction_id,
                "created_at": data.get("created_at"),
                "threshold": data.get("threshold"),
                "horizon": data.get("horizon"),
                "grid_size": data.get("grid_size"),
                "processing_time_ms": data.get("processing_time_ms"),
                "prediction_engine": data.get("prediction_engine", data.get("model_backend")),
                "model_backend": data.get("model_backend", data.get("prediction_engine")),
                "preprocess_mode": data.get("preprocess_mode"),
                "model_profile": data.get("model_profile"),
                "inference_mode": data.get("inference_mode"),
                "recommended_threshold": data.get("recommended_threshold"),
                "output_shape": data.get("output_shape"),
                "input_dates": data.get("input_dates"),
                "target_date": data.get("target_date"),
                "admin_regions": data.get("admin_regions", get_admin_region_legend()),
                "outputs": data.get("outputs", []),
                "detail_url": f"/api/predictions/{prediction_id}",
                "download_zip_url": f"/api/predictions/{prediction_id}/download",
            }
        )

        # Menghentikan proses jika jumlah record sudah mencapai limit.
        if len(records) >= limit:
            break

    # Mengembalikan daftar riwayat prediksi.
    return records


# Endpoint untuk mengambil detail satu prediksi berdasarkan prediction_id.
@router.get("/predictions/{prediction_id}")
def get_prediction_detail(prediction_id: str) -> dict:
    # Membaca metadata prediksi atau mengembalikan 404 jika tidak ditemukan.
    data, input_dir, output_dir = _load_prediction_or_404(prediction_id)

    # Mengembalikan detail prediksi yang sudah diperkaya URL input/output.
    return _enrich_prediction_detail(
        data=data,
        prediction_id=prediction_id,
        input_dir=input_dir,
        output_dir=output_dir,
    )


# Endpoint untuk mengunduh semua input dan output prediksi dalam format ZIP.
@router.get("/predictions/{prediction_id}/download")
def download_prediction_bundle(prediction_id: str) -> FileResponse:
    # Membaca folder prediksi atau mengembalikan 404 jika tidak ditemukan.
    _, input_dir, output_dir = _load_prediction_or_404(prediction_id)

    # Menentukan path file ZIP yang akan dibuat.
    zip_path = output_dir / f"{prediction_id}_bundle.zip"

    # Membuat ZIP berisi file input dan output prediksi.
    _build_prediction_zip(zip_path=zip_path, input_dir=input_dir, output_dir=output_dir)

    # Mengembalikan file ZIP sebagai response download.
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"prediction_{prediction_id}.zip",
    )


# Endpoint untuk mengecek status runtime model dan konfigurasi aktif.
@router.get("/runtime/status")
def runtime_status(request: Request) -> dict:
    # Mengambil service inferensi dari app.state.
    inference_service = request.app.state.inference_service

    # Menyiapkan nilai versi TensorFlow; None jika tidak tersedia.
    tensorflow_version = None

    # Mencoba membaca versi TensorFlow pada environment runtime.
    try:
        # Import TensorFlow dilakukan di dalam try agar aplikasi tetap berjalan jika TensorFlow belum terinstall.
        import tensorflow as tf  # type: ignore

        # Menyimpan versi TensorFlow jika import berhasil.
        tensorflow_version = tf.__version__

    # Jika TensorFlow tidak tersedia atau gagal diimport, versi tetap None.
    except Exception:
        tensorflow_version = None

    # Mengembalikan status runtime untuk ditampilkan pada dashboard.
    return {
        "prediction_engine": inference_service.backend,
        "model_backend": inference_service.backend,
        "model_path": inference_service.model_path,
        "model_profile": request.app.state.model_spec.profile_name,
        "model_display_name": request.app.state.model_spec.display_name,
        "inference_mode": request.app.state.model_spec.inference_mode,
        "grid_size": request.app.state.model_spec.grid_size,
        "channels": request.app.state.model_spec.channels,
        "time_steps": request.app.state.model_spec.time_steps,
        "patch_size": request.app.state.model_spec.patch_size,
        "patch_stride": request.app.state.model_spec.patch_stride,
        "input_dilation_kernel": request.app.state.model_spec.input_dilation_kernel,
        "recommended_threshold": request.app.state.model_spec.recommended_threshold,
        "preprocess_mode": inference_service.preprocess_mode,
        "tensorflow_version": tensorflow_version,
        "model_candidates": inference_service.list_model_candidates(),
    }
