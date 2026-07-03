# File anotasi dari `backend/routes/predict.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Endpoint API prediksi, riwayat, detail, dan unduhan hasil sistem web.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from datetime import datetime, timezone
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from time import perf_counter
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from uuid import uuid4
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import zipfile

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi.responses import FileResponse
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pydantic import BaseModel

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend import settings
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.services.postprocess import (
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    save_binary_mask,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    save_heatmap,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    save_overlay,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    save_probability_array,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.services.preprocess import build_hotspot_mask_sequence, build_input_tensor
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.services.storage import (
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    create_prediction_dirs,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    read_json,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    save_upload_files,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    to_public_storage_url,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    write_json,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
)
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.services.validator import validate_and_order_files

# Menyimpan nilai ke `router` untuk dipakai pada langkah berikutnya.
router = APIRouter(prefix="/api", tags=["prediction"])


# Membuat wadah bernama `PredictionOutput` untuk menyimpan data atau aturan kerja.
class PredictionOutput(BaseModel):
    # Menjelaskan data `step` yang disimpan atau dikirim pada bagian ini.
    step: int
    # Menjelaskan data `heatmap_url` yang disimpan atau dikirim pada bagian ini.
    heatmap_url: str
    # Menyimpan nilai ke `overlay_url` untuk dipakai pada langkah berikutnya.
    overlay_url: str | None = None
    # Menyimpan nilai ke `binary_url` untuk dipakai pada langkah berikutnya.
    binary_url: str | None = None


# Membuat wadah bernama `PredictResponse` untuk menyimpan data atau aturan kerja.
class PredictResponse(BaseModel):
    # Menjelaskan data `prediction_id` yang disimpan atau dikirim pada bagian ini.
    prediction_id: str
    # Menjelaskan data `heatmap_url` yang disimpan atau dikirim pada bagian ini.
    heatmap_url: str
    # Menyimpan nilai ke `overlay_url` untuk dipakai pada langkah berikutnya.
    overlay_url: str | None = None
    # Menyimpan nilai ke `download_zip_url` untuk dipakai pada langkah berikutnya.
    download_zip_url: str | None = None
    # Menjelaskan data `processing_time_ms` yang disimpan atau dikirim pada bagian ini.
    processing_time_ms: int
    # Menjelaskan data `grid_size` yang disimpan atau dikirim pada bagian ini.
    grid_size: int
    # Menyimpan nilai ke `threshold` untuk dipakai pada langkah berikutnya.
    threshold: float | None = None
    # Menjelaskan data `horizon` yang disimpan atau dikirim pada bagian ini.
    horizon: int
    # Menjelaskan data `prediction_engine` yang disimpan atau dikirim pada bagian ini.
    prediction_engine: str
    # Menyimpan nilai ke `model_backend` untuk dipakai pada langkah berikutnya.
    model_backend: str | None = None
    # Menyimpan nilai ke `preprocess_mode` untuk dipakai pada langkah berikutnya.
    preprocess_mode: str | None = None
    # Menyimpan nilai ke `model_profile` untuk dipakai pada langkah berikutnya.
    model_profile: str | None = None
    # Menyimpan nilai ke `inference_mode` untuk dipakai pada langkah berikutnya.
    inference_mode: str | None = None
    # Menyimpan nilai ke `model_path` untuk dipakai pada langkah berikutnya.
    model_path: str | None = None
    # Menyimpan nilai ke `recommended_threshold` untuk dipakai pada langkah berikutnya.
    recommended_threshold: float | None = None
    # Menyimpan nilai ke `output_shape` untuk dipakai pada langkah berikutnya.
    output_shape: list[int] | None = None
    # Menjelaskan data `outputs` yang disimpan atau dikirim pada bagian ini.
    outputs: list[PredictionOutput]


# Membuat langkah kerja bernama `_load_prediction_or_404`.
def _load_prediction_or_404(prediction_id: str) -> tuple[dict, Path, Path]:
    # Menyimpan nilai ke `input_dir` untuk dipakai pada langkah berikutnya.
    input_dir = settings.INPUTS_DIR / prediction_id
    # Menyimpan nilai ke `output_dir` untuk dipakai pada langkah berikutnya.
    output_dir = settings.OUTPUTS_DIR / prediction_id
    # Menyimpan nilai ke `params_path` untuk dipakai pada langkah berikutnya.
    params_path = output_dir / "params.json"
    # Mengecek syarat sebelum melanjutkan proses.
    if not params_path.exists():
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise HTTPException(
            # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
            status_code=status.HTTP_404_NOT_FOUND,
            # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
            detail=f"Prediction ID '{prediction_id}' tidak ditemukan.",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return read_json(params_path), input_dir, output_dir


# Membuat langkah kerja bernama `_build_prediction_zip`.
def _build_prediction_zip(zip_path: Path, input_dir: Path, output_dir: Path) -> None:
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Mengulang proses untuk setiap data dalam daftar.
        for path in sorted(input_dir.glob("*")):
            # Mengecek syarat sebelum melanjutkan proses.
            if path.is_file():
                # Menyimpan nilai ke `archive.write(path, arcname` untuk dipakai pada langkah berikutnya.
                archive.write(path, arcname=f"inputs/{path.name}")

        # Mengulang proses untuk setiap data dalam daftar.
        for path in sorted(output_dir.glob("*")):
            # Mengecek syarat sebelum melanjutkan proses.
            if not path.is_file():
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Mengecek syarat sebelum melanjutkan proses.
            if path == zip_path:
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Menyimpan nilai ke `archive.write(path, arcname` untuk dipakai pada langkah berikutnya.
            archive.write(path, arcname=f"outputs/{path.name}")


# Membuat langkah kerja bernama `_enrich_prediction_detail`.
def _enrich_prediction_detail(data: dict, prediction_id: str, input_dir: Path) -> dict:
    # Menyimpan nilai ke `payload` untuk dipakai pada langkah berikutnya.
    payload = dict(data)

    # Menyimpan nilai ke `outputs` untuk dipakai pada langkah berikutnya.
    outputs = payload.get("outputs", [])
    # Mengecek syarat sebelum melanjutkan proses.
    if isinstance(outputs, list) and outputs:
        # Menyimpan nilai ke `first` untuk dipakai pada langkah berikutnya.
        first = outputs[0]
        # Mengecek syarat sebelum melanjutkan proses.
        if isinstance(first, dict):
            # Melanjutkan langkah kerja pada bagian kode ini.
            payload.setdefault("heatmap_url", first.get("heatmap_url"))
            # Melanjutkan langkah kerja pada bagian kode ini.
            payload.setdefault("overlay_url", first.get("overlay_url"))

    # Menyimpan nilai ke `input_files` untuk dipakai pada langkah berikutnya.
    input_files = payload.get("input_files", {})
    # Menyimpan nilai ke `input_urls` untuk dipakai pada langkah berikutnya.
    input_urls: dict[str, str] = {}
    # Mengecek syarat sebelum melanjutkan proses.
    if isinstance(input_files, dict):
        # Mengulang proses untuk setiap data dalam daftar.
        for stem, file_name in input_files.items():
            # Mengecek syarat sebelum melanjutkan proses.
            if not isinstance(file_name, str):
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Menyimpan nilai ke `file_path` untuk dipakai pada langkah berikutnya.
            file_path = input_dir / file_name
            # Mengecek syarat sebelum melanjutkan proses.
            if file_path.exists():
                # Menyimpan nilai ke `input_urls[stem]` untuk dipakai pada langkah berikutnya.
                input_urls[stem] = to_public_storage_url(file_path)

    # Menyimpan nilai ke `payload["input_urls"]` untuk dipakai pada langkah berikutnya.
    payload["input_urls"] = input_urls
    # Menyimpan nilai ke `payload["detail_url"]` untuk dipakai pada langkah berikutnya.
    payload["detail_url"] = f"/api/predictions/{prediction_id}"
    # Menyimpan nilai ke `payload["download_zip_url"]` untuk dipakai pada langkah berikutnya.
    payload["download_zip_url"] = f"/api/predictions/{prediction_id}/download"
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return payload


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@router.post("/predict", response_model=PredictResponse)
# Membuat langkah kerja `predict` yang bisa menunggu proses web/backend selesai.
async def predict(
    # Menjelaskan data `request` yang disimpan atau dikirim pada bagian ini.
    request: Request,
    # Menyimpan nilai ke `files` untuk dipakai pada langkah berikutnya.
    files: list[UploadFile] = File(...),
    # Menyimpan nilai ke `threshold` untuk dipakai pada langkah berikutnya.
    threshold: float | None = Form(default=None),
    # Menyimpan nilai ke `horizon` untuk dipakai pada langkah berikutnya.
    horizon: int = Form(default=1),
# Menutup susunan data atau perintah yang dimulai sebelumnya.
) -> PredictResponse:
    # Mengecek syarat sebelum melanjutkan proses.
    if threshold is not None and not (0.0 <= threshold <= 1.0):
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise HTTPException(
            # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
            detail="Nilai threshold harus pada rentang 0 sampai 1.",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Mengecek syarat sebelum melanjutkan proses.
    if not (1 <= horizon <= 7):
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise HTTPException(
            # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
            detail="Nilai horizon harus pada rentang 1 sampai 7.",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Menyimpan nilai ke `ordered_files` untuk dipakai pada langkah berikutnya.
    ordered_files = await validate_and_order_files(
        # Menyimpan nilai ke `files` untuk dipakai pada langkah berikutnya.
        files=files,
        # Menyimpan nilai ke `required_stems` untuk dipakai pada langkah berikutnya.
        required_stems=settings.REQUIRED_STEMS,
        # Menyimpan nilai ke `allowed_extensions` untuk dipakai pada langkah berikutnya.
        allowed_extensions=settings.ALLOWED_EXTENSIONS,
        # Menyimpan nilai ke `max_file_bytes` untuk dipakai pada langkah berikutnya.
        max_file_bytes=settings.MAX_FILE_BYTES,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `prediction_id` untuk dipakai pada langkah berikutnya.
    prediction_id = str(uuid4())
    # Menyimpan nilai ke `input_dir, output_dir` untuk dipakai pada langkah berikutnya.
    input_dir, output_dir = create_prediction_dirs(prediction_id)

    # Menyimpan nilai ke `saved_inputs` untuk dipakai pada langkah berikutnya.
    saved_inputs = await save_upload_files(ordered_files=ordered_files, input_dir=input_dir)
    # Menyimpan nilai ke `metadata_inputs` untuk dipakai pada langkah berikutnya.
    metadata_inputs = {stem: path.name for stem, path in saved_inputs.items()}

    # Menyimpan nilai ke `model_spec` untuk dipakai pada langkah berikutnya.
    model_spec = request.app.state.model_spec
    # Menyimpan nilai ke `inference_service` untuk dipakai pada langkah berikutnya.
    inference_service = request.app.state.inference_service

    # Menyimpan nilai ke `start` untuk dipakai pada langkah berikutnya.
    start = perf_counter()
    # Mengecek syarat sebelum melanjutkan proses.
    if model_spec.inference_mode == "patch_stitch":
        # Menyimpan nilai ke `sequence` untuk dipakai pada langkah berikutnya.
        sequence = build_hotspot_mask_sequence(
            # Menyimpan nilai ke `ordered_input_paths` untuk dipakai pada langkah berikutnya.
            ordered_input_paths=saved_inputs,
            # Menyimpan nilai ke `required_stems` untuk dipakai pada langkah berikutnya.
            required_stems=settings.REQUIRED_STEMS,
            # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
            input_dilation_kernel=model_spec.input_dilation_kernel,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `predicted_maps` untuk dipakai pada langkah berikutnya.
        predicted_maps = inference_service.predict_horizon(sequence=sequence, horizon=horizon)
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menyimpan nilai ke `tensor` untuk dipakai pada langkah berikutnya.
        tensor = build_input_tensor(
            # Menyimpan nilai ke `ordered_input_paths` untuk dipakai pada langkah berikutnya.
            ordered_input_paths=saved_inputs,
            # Menyimpan nilai ke `required_stems` untuk dipakai pada langkah berikutnya.
            required_stems=settings.REQUIRED_STEMS,
            # Menyimpan nilai ke `spec` untuk dipakai pada langkah berikutnya.
            spec=model_spec,
            # Menyimpan nilai ke `preprocess_mode` untuk dipakai pada langkah berikutnya.
            preprocess_mode=inference_service.preprocess_mode,
            # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
            input_dilation_kernel=model_spec.input_dilation_kernel,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `predicted_maps` untuk dipakai pada langkah berikutnya.
        predicted_maps = inference_service.predict_horizon(sequence=tensor, horizon=horizon)
    # Menyimpan nilai ke `duration_ms` untuk dipakai pada langkah berikutnya.
    duration_ms = int((perf_counter() - start) * 1000)

    # Menyimpan nilai ke `reference_h0` untuk dipakai pada langkah berikutnya.
    reference_h0 = saved_inputs["H0"]
    # Menyimpan nilai ke `outputs` untuk dipakai pada langkah berikutnya.
    outputs: list[PredictionOutput] = []
    # Menyimpan nilai ke `outputs_metadata` untuk dipakai pada langkah berikutnya.
    outputs_metadata: list[dict] = []

    # Mengulang proses untuk setiap data dalam daftar.
    for index, probability in enumerate(predicted_maps, start=1):
        # Menyimpan nilai ke `step_name` untuk dipakai pada langkah berikutnya.
        step_name = f"H+{index}"
        # Menyimpan nilai ke `prob_path` untuk dipakai pada langkah berikutnya.
        prob_path = output_dir / f"probability_{step_name}.npy"
        # Menyimpan nilai ke `heatmap_path` untuk dipakai pada langkah berikutnya.
        heatmap_path = output_dir / f"heatmap_{step_name}.png"
        # Menyimpan nilai ke `overlay_path` untuk dipakai pada langkah berikutnya.
        overlay_path = output_dir / f"overlay_{step_name}.png"

        # Menyimpan nilai ke `save_probability_array(probability` untuk dipakai pada langkah berikutnya.
        save_probability_array(probability=probability, path=prob_path)
        # Menyimpan nilai ke `save_heatmap(probability` untuk dipakai pada langkah berikutnya.
        save_heatmap(probability=probability, path=heatmap_path)
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        save_overlay(
            # Menyimpan nilai ke `probability` untuk dipakai pada langkah berikutnya.
            probability=probability,
            # Menyimpan nilai ke `reference_input` untuk dipakai pada langkah berikutnya.
            reference_input=reference_h0,
            # Menyimpan nilai ke `destination_path` untuk dipakai pada langkah berikutnya.
            destination_path=overlay_path,
            # Menyimpan nilai ke `base_map_path` untuk dipakai pada langkah berikutnya.
            base_map_path=settings.BASE_MAP_PATH,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

        # Menyimpan nilai ke `binary_url` untuk dipakai pada langkah berikutnya.
        binary_url = None
        # Menyimpan nilai ke `binary_path` untuk dipakai pada langkah berikutnya.
        binary_path = None
        # Mengecek syarat sebelum melanjutkan proses.
        if threshold is not None:
            # Menyimpan nilai ke `binary_path` untuk dipakai pada langkah berikutnya.
            binary_path = output_dir / f"binary_{step_name}.png"
            # Menyimpan nilai ke `save_binary_mask(probability` untuk dipakai pada langkah berikutnya.
            save_binary_mask(probability=probability, threshold=threshold, path=binary_path)
            # Menyimpan nilai ke `binary_url` untuk dipakai pada langkah berikutnya.
            binary_url = to_public_storage_url(binary_path)

        # Menyimpan nilai ke `output_entry` untuk dipakai pada langkah berikutnya.
        output_entry = PredictionOutput(
            # Menyimpan nilai ke `step` untuk dipakai pada langkah berikutnya.
            step=index,
            # Menyimpan nilai ke `heatmap_url` untuk dipakai pada langkah berikutnya.
            heatmap_url=to_public_storage_url(heatmap_path),
            # Menyimpan nilai ke `overlay_url` untuk dipakai pada langkah berikutnya.
            overlay_url=to_public_storage_url(overlay_path),
            # Menyimpan nilai ke `binary_url` untuk dipakai pada langkah berikutnya.
            binary_url=binary_url,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Melanjutkan langkah kerja pada bagian kode ini.
        outputs.append(output_entry)
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        outputs_metadata.append(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                # Melanjutkan langkah kerja pada bagian kode ini.
                "step": index,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "heatmap_url": output_entry.heatmap_url,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "overlay_url": output_entry.overlay_url,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "binary_url": output_entry.binary_url,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "probability_file": prob_path.name,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "heatmap_file": heatmap_path.name,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "overlay_file": overlay_path.name,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "binary_file": binary_path.name if binary_path else None,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            }
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Menyimpan nilai ke `metadata` untuk dipakai pada langkah berikutnya.
    metadata = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "prediction_id": prediction_id,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "threshold": threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recommended_threshold": model_spec.recommended_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "horizon": horizon,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "grid_size": model_spec.grid_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "channels": model_spec.channels,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_profile": model_spec.profile_name,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_display_name": model_spec.display_name,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "inference_mode": model_spec.inference_mode,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_size": model_spec.patch_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_stride": model_spec.patch_stride,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_dilation_kernel": model_spec.input_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "prediction_engine": inference_service.backend,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_backend": inference_service.backend,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_path": inference_service.model_path,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "processing_time_ms": duration_ms,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "preprocess_mode": inference_service.preprocess_mode,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "output_shape": list(predicted_maps[0].shape) if predicted_maps else None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_files": metadata_inputs,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "outputs": outputs_metadata,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Melanjutkan langkah kerja pada bagian kode ini.
    write_json(input_dir / "params.json", metadata)
    # Melanjutkan langkah kerja pada bagian kode ini.
    write_json(output_dir / "params.json", metadata)

    # Menyimpan nilai ke `first` untuk dipakai pada langkah berikutnya.
    first = outputs[0]
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return PredictResponse(
        # Menyimpan nilai ke `prediction_id` untuk dipakai pada langkah berikutnya.
        prediction_id=prediction_id,
        # Menyimpan nilai ke `heatmap_url` untuk dipakai pada langkah berikutnya.
        heatmap_url=first.heatmap_url,
        # Menyimpan nilai ke `overlay_url` untuk dipakai pada langkah berikutnya.
        overlay_url=first.overlay_url,
        # Menyimpan nilai ke `download_zip_url` untuk dipakai pada langkah berikutnya.
        download_zip_url=f"/api/predictions/{prediction_id}/download",
        # Menyimpan nilai ke `processing_time_ms` untuk dipakai pada langkah berikutnya.
        processing_time_ms=duration_ms,
        # Menyimpan nilai ke `grid_size` untuk dipakai pada langkah berikutnya.
        grid_size=model_spec.grid_size,
        # Menyimpan nilai ke `threshold` untuk dipakai pada langkah berikutnya.
        threshold=threshold,
        # Menyimpan nilai ke `horizon` untuk dipakai pada langkah berikutnya.
        horizon=horizon,
        # Menyimpan nilai ke `prediction_engine` untuk dipakai pada langkah berikutnya.
        prediction_engine=inference_service.backend,
        # Menyimpan nilai ke `model_backend` untuk dipakai pada langkah berikutnya.
        model_backend=inference_service.backend,
        # Menyimpan nilai ke `preprocess_mode` untuk dipakai pada langkah berikutnya.
        preprocess_mode=inference_service.preprocess_mode,
        # Menyimpan nilai ke `model_profile` untuk dipakai pada langkah berikutnya.
        model_profile=model_spec.profile_name,
        # Menyimpan nilai ke `inference_mode` untuk dipakai pada langkah berikutnya.
        inference_mode=model_spec.inference_mode,
        # Menyimpan nilai ke `model_path` untuk dipakai pada langkah berikutnya.
        model_path=inference_service.model_path,
        # Menyimpan nilai ke `recommended_threshold` untuk dipakai pada langkah berikutnya.
        recommended_threshold=model_spec.recommended_threshold,
        # Menyimpan nilai ke `output_shape` untuk dipakai pada langkah berikutnya.
        output_shape=list(predicted_maps[0].shape) if predicted_maps else None,
        # Menyimpan nilai ke `outputs` untuk dipakai pada langkah berikutnya.
        outputs=outputs,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@router.get("/predictions")
# Membuat langkah kerja bernama `list_predictions`.
def list_predictions(limit: int = 20) -> list[dict]:
    # Mengecek syarat sebelum melanjutkan proses.
    if limit < 1:
        # Menghentikan proses dan memberi pesan kesalahan yang jelas.
        raise HTTPException(
            # Menyimpan nilai ke `status_code` untuk dipakai pada langkah berikutnya.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
            detail="Parameter limit minimal 1.",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Menyimpan nilai ke `params_files` untuk dipakai pada langkah berikutnya.
    params_files = sorted(
        # Melanjutkan langkah kerja pada bagian kode ini.
        settings.OUTPUTS_DIR.glob("*/params.json"),
        # Menyimpan nilai ke `key` untuk dipakai pada langkah berikutnya.
        key=lambda path: path.stat().st_mtime,
        # Menyimpan nilai ke `reverse` untuk dipakai pada langkah berikutnya.
        reverse=True,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `records` untuk dipakai pada langkah berikutnya.
    records: list[dict] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for params_file in params_files[:limit]:
        # Menyimpan nilai ke `data` untuk dipakai pada langkah berikutnya.
        data = read_json(params_file)
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        records.append(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                # Melanjutkan langkah kerja pada bagian kode ini.
                "prediction_id": data.get("prediction_id"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "created_at": data.get("created_at"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "threshold": data.get("threshold"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "horizon": data.get("horizon"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "grid_size": data.get("grid_size"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "processing_time_ms": data.get("processing_time_ms"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "prediction_engine": data.get("prediction_engine", data.get("model_backend")),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "model_backend": data.get("model_backend", data.get("prediction_engine")),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "preprocess_mode": data.get("preprocess_mode"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "model_profile": data.get("model_profile"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "inference_mode": data.get("inference_mode"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "recommended_threshold": data.get("recommended_threshold"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "output_shape": data.get("output_shape"),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "outputs": data.get("outputs", []),
                # Melanjutkan langkah kerja pada bagian kode ini.
                "detail_url": f"/api/predictions/{data.get('prediction_id')}",
                # Melanjutkan langkah kerja pada bagian kode ini.
                "download_zip_url": f"/api/predictions/{data.get('prediction_id')}/download",
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            }
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return records


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@router.get("/predictions/{prediction_id}")
# Membuat langkah kerja bernama `get_prediction_detail`.
def get_prediction_detail(prediction_id: str) -> dict:
    # Menyimpan nilai ke `data, input_dir, _` untuk dipakai pada langkah berikutnya.
    data, input_dir, _ = _load_prediction_or_404(prediction_id)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return _enrich_prediction_detail(data=data, prediction_id=prediction_id, input_dir=input_dir)


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@router.get("/predictions/{prediction_id}/download")
# Membuat langkah kerja bernama `download_prediction_bundle`.
def download_prediction_bundle(prediction_id: str) -> FileResponse:
    # Menyimpan nilai ke `_, input_dir, output_dir` untuk dipakai pada langkah berikutnya.
    _, input_dir, output_dir = _load_prediction_or_404(prediction_id)
    # Menyimpan nilai ke `zip_path` untuk dipakai pada langkah berikutnya.
    zip_path = output_dir / f"{prediction_id}_bundle.zip"
    # Menyimpan nilai ke `_build_prediction_zip(zip_path` untuk dipakai pada langkah berikutnya.
    _build_prediction_zip(zip_path=zip_path, input_dir=input_dir, output_dir=output_dir)

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return FileResponse(
        # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
        path=zip_path,
        # Menyimpan nilai ke `media_type` untuk dipakai pada langkah berikutnya.
        media_type="application/zip",
        # Menyimpan nilai ke `filename` untuk dipakai pada langkah berikutnya.
        filename=f"prediction_{prediction_id}.zip",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@router.get("/runtime/status")
# Membuat langkah kerja bernama `runtime_status`.
def runtime_status(request: Request) -> dict:
    # Menyimpan nilai ke `inference_service` untuk dipakai pada langkah berikutnya.
    inference_service = request.app.state.inference_service
    # Menyimpan nilai ke `tensorflow_version` untuk dipakai pada langkah berikutnya.
    tensorflow_version = None
    # Mencoba menjalankan proses yang mungkin gagal.
    try:
        # Mengambil alat bantu/library yang diperlukan oleh file ini.
        import tensorflow as tf  # type: ignore

        # Menyimpan nilai ke `tensorflow_version` untuk dipakai pada langkah berikutnya.
        tensorflow_version = tf.__version__
    # Menangani kesalahan agar program tidak langsung berhenti.
    except Exception:
        # Menyimpan nilai ke `tensorflow_version` untuk dipakai pada langkah berikutnya.
        tensorflow_version = None

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "prediction_engine": inference_service.backend,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_backend": inference_service.backend,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_path": inference_service.model_path,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_profile": request.app.state.model_spec.profile_name,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_display_name": request.app.state.model_spec.display_name,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "inference_mode": request.app.state.model_spec.inference_mode,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "grid_size": request.app.state.model_spec.grid_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "channels": request.app.state.model_spec.channels,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "time_steps": request.app.state.model_spec.time_steps,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_size": request.app.state.model_spec.patch_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_stride": request.app.state.model_spec.patch_stride,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_dilation_kernel": request.app.state.model_spec.input_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recommended_threshold": request.app.state.model_spec.recommended_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "preprocess_mode": inference_service.preprocess_mode,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tensorflow_version": tensorflow_version,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_candidates": inference_service.list_model_candidates(),
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
