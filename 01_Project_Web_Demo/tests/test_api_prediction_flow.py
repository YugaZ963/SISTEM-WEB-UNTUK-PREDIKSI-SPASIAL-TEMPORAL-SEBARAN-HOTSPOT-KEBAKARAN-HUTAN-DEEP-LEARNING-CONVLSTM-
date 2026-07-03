
"""Komentar file skripsi:
Pengujian API untuk memastikan alur upload, prediksi, dan output web berjalan.

Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
terutama data understanding, data preparation, modeling, evaluation,
atau deployment sesuai fungsi teknisnya.
"""

from __future__ import annotations

# io dipakai di test untuk membuat file gambar sementara di memori tanpa menulis file manual.
import io
# zipfile dipakai untuk memeriksa isi bundle hasil prediksi pada test API.
import zipfile
# timedelta dipakai untuk menghitung tanggal target dari H0 + 1 hari.
from datetime import date, timedelta
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path

# TestClient menjalankan aplikasi FastAPI secara lokal saat test endpoint prediksi.
from fastapi.testclient import TestClient
# Pillow Image dipakai untuk membuka citra hotspot PNG/JPG dan menyimpan preview/hasil.
from PIL import Image

from backend import settings
from backend.main import create_app


# Membuat citra PNG kecil untuk simulasi upload pada test API.
def _make_png_bytes(rgb: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (32, 32), rgb)
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    # Hasil ini dikembalikan sebagai output fungsi `_make_png_bytes` untuk tahap berikutnya.
    return payload.getvalue()


# Menyusun tujuh file upload H-6 sampai H0 yang valid untuk test endpoint prediksi.
def _valid_upload_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    colors = [(255, 0, 0), (200, 0, 0), (150, 0, 0), (100, 0, 0), (50, 0, 0), (25, 0, 0), (10, 0, 0)]
    for stem, color in zip(settings.REQUIRED_STEMS, colors):
        files.append(("files", (f"{stem}.png", _make_png_bytes(color), "image/png")))
    # Hasil ini dikembalikan sebagai output fungsi `_valid_upload_files` untuk tahap berikutnya.
    return files


# Menyusun tujuh file upload bertanggal agar test bisa memeriksa target_date H+1.
def _valid_upload_files_with_dates(start_date: str = "2025-07-13") -> list[tuple[str, tuple[str, bytes, str]]]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    colors = [(255, 0, 0), (200, 0, 0), (150, 0, 0), (100, 0, 0), (50, 0, 0), (25, 0, 0), (10, 0, 0)]
    first_date = date.fromisoformat(start_date)
    for offset, (stem, color) in enumerate(zip(settings.REQUIRED_STEMS, colors)):
        day = first_date + timedelta(days=offset)
        files.append(("files", (f"{stem}_{day.isoformat()}.png", _make_png_bytes(color), "image/png")))
    # Hasil ini dikembalikan sebagai output fungsi `_valid_upload_files_with_dates` untuk tahap berikutnya.
    return files


# Membuat aplikasi FastAPI test dengan folder storage sementara agar test tidak mengotori data asli.
def _create_isolated_app(tmp_path: Path) -> TestClient:
    # `storage_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    storage_dir = tmp_path / "storage"
    # `inputs_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    inputs_dir = storage_dir / "inputs"
    # `outputs_dir` menyimpan folder kerja untuk dataset, artefak, atau output proses.
    outputs_dir = storage_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    settings.STORAGE_DIR = storage_dir
    settings.INPUTS_DIR = inputs_dir
    settings.OUTPUTS_DIR = outputs_dir

    app = create_app()
    # Hasil ini dikembalikan sebagai output fungsi `_create_isolated_app` untuk tahap berikutnya.
    return TestClient(app)


# Fungsi `test_prediction_detail_and_zip_flow` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def test_prediction_detail_and_zip_flow(tmp_path: Path) -> None:
    client = _create_isolated_app(tmp_path)

    response = client.post(
        "/api/predict",
        files=_valid_upload_files(),
        data={"threshold": "0.5", "horizon": "1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["horizon"] == 1
    assert len(payload["outputs"]) == 1
    prediction_id = payload["prediction_id"]

    detail = client.get(f"/api/predictions/{prediction_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["prediction_id"] == prediction_id
    assert detail_payload["download_zip_url"].endswith("/download")
    assert "H0" in detail_payload["input_urls"]

    zip_response = client.get(f"/api/predictions/{prediction_id}/download")
    assert zip_response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(zip_response.content))
    names = set(archive.namelist())
    assert any(name.startswith("inputs/H0.") for name in names)
    assert "outputs/params.json" in names
    assert any(name.startswith("outputs/heatmap_H+1") for name in names)


# Fungsi `test_predict_accepts_dated_sequence_and_returns_target_date` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def test_predict_accepts_dated_sequence_and_returns_target_date(tmp_path: Path) -> None:
    client = _create_isolated_app(tmp_path)

    response = client.post(
        "/api/predict",
        files=_valid_upload_files_with_dates(),
        data={"threshold": "0.5", "horizon": "1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["input_dates"]["H-6"] == "2025-07-13"
    assert payload["input_dates"]["H0"] == "2025-07-19"
    assert payload["target_date"] == "2025-07-20"


# Fungsi `test_predict_rejects_partial_dated_sequence` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def test_predict_rejects_partial_dated_sequence(tmp_path: Path) -> None:
    client = _create_isolated_app(tmp_path)
    files = _valid_upload_files_with_dates()
    files[0] = ("files", ("H-6.png", files[0][1][1], "image/png"))

    response = client.post(
        "/api/predict",
        files=files,
        data={"threshold": "0.5", "horizon": "1"},
    )
    assert response.status_code == 400
    assert "semua file H-6 sampai H0 harus memakai tanggal" in response.json()["detail"]


# Fungsi `test_predict_rejects_wrong_daily_date_order` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def test_predict_rejects_wrong_daily_date_order(tmp_path: Path) -> None:
    client = _create_isolated_app(tmp_path)
    files = _valid_upload_files_with_dates()
    files[1] = ("files", ("H-5_2025-07-15.png", files[1][1][1], "image/png"))

    response = client.post(
        "/api/predict",
        files=files,
        data={"threshold": "0.5", "horizon": "1"},
    )
    assert response.status_code == 400
    assert "Tanggal file H-5 harus 2025-07-14" in response.json()["detail"]


# Fungsi `test_predict_reject_horizon_more_than_h1` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def test_predict_reject_horizon_more_than_h1(tmp_path: Path) -> None:
    client = _create_isolated_app(tmp_path)

    response = client.post(
        "/api/predict",
        files=_valid_upload_files(),
        data={"threshold": "0.5", "horizon": "2"},
    )
    assert response.status_code == 400
    assert "hanya mendukung prediksi H+1" in response.json()["detail"]


# Fungsi `test_predict_reject_invalid_file_count` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def test_predict_reject_invalid_file_count(tmp_path: Path) -> None:
    client = _create_isolated_app(tmp_path)
    short_files = _valid_upload_files()[:6]

    response = client.post("/api/predict", files=short_files)
    assert response.status_code == 400
    assert "7 citra berurutan" in response.json()["detail"]


# Fungsi `test_predict_reject_non_image_mime` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def test_predict_reject_non_image_mime(tmp_path: Path) -> None:
    client = _create_isolated_app(tmp_path)
    files = _valid_upload_files()
    files[0] = ("files", ("H-6.png", b"not-an-image", "text/plain"))

    response = client.post("/api/predict", files=files)
    assert response.status_code == 400
    assert "MIME type" in response.json()["detail"]


# Fungsi `test_runtime_status_endpoint` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def test_runtime_status_endpoint(tmp_path: Path) -> None:
    client = _create_isolated_app(tmp_path)
    response = client.get("/api/runtime/status")
    assert response.status_code == 200
    payload = response.json()
    assert "prediction_engine" in payload
    assert "model_candidates" in payload
    assert isinstance(payload["model_candidates"], list)
