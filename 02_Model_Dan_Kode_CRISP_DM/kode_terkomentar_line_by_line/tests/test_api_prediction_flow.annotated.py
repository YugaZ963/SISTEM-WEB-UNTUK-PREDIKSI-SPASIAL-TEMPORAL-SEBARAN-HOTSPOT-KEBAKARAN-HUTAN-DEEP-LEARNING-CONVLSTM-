# File anotasi dari `tests/test_api_prediction_flow.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Pengujian sistem, yaitu memastikan fitur berjalan benar.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Pengujian API untuk memastikan alur upload, prediksi, dan output web berjalan.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Menjaga cara baca kode Python tetap cocok dengan versi yang digunakan.
from __future__ import annotations

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import io
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import zipfile
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi.testclient import TestClient
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from PIL import Image

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend import settings
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.main import create_app


# Membuat langkah kerja bernama `_make_png_bytes`.
def _make_png_bytes(rgb: tuple[int, int, int]) -> bytes:
    # Menyimpan nilai ke `image` untuk dipakai pada langkah berikutnya.
    image = Image.new("RGB", (32, 32), rgb)
    # Menyimpan nilai ke `payload` untuk dipakai pada langkah berikutnya.
    payload = io.BytesIO()
    # Menyimpan nilai ke `image.save(payload, format` untuk dipakai pada langkah berikutnya.
    image.save(payload, format="PNG")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return payload.getvalue()


# Membuat langkah kerja bernama `_valid_upload_files`.
def _valid_upload_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    # Menyimpan nilai ke `files` untuk dipakai pada langkah berikutnya.
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    # Menyimpan nilai ke `colors` untuk dipakai pada langkah berikutnya.
    colors = [(255, 0, 0), (200, 0, 0), (150, 0, 0), (100, 0, 0), (50, 0, 0), (25, 0, 0), (10, 0, 0)]
    # Mengulang proses untuk setiap data dalam daftar.
    for stem, color in zip(settings.REQUIRED_STEMS, colors):
        # Melanjutkan langkah kerja pada bagian kode ini.
        files.append(("files", (f"{stem}.png", _make_png_bytes(color), "image/png")))
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return files


# Membuat langkah kerja bernama `_create_isolated_app`.
def _create_isolated_app(tmp_path: Path) -> TestClient:
    # Menyimpan nilai ke `storage_dir` untuk dipakai pada langkah berikutnya.
    storage_dir = tmp_path / "storage"
    # Menyimpan nilai ke `inputs_dir` untuk dipakai pada langkah berikutnya.
    inputs_dir = storage_dir / "inputs"
    # Menyimpan nilai ke `outputs_dir` untuk dipakai pada langkah berikutnya.
    outputs_dir = storage_dir / "outputs"
    # Menyimpan nilai ke `inputs_dir.mkdir(parents` untuk dipakai pada langkah berikutnya.
    inputs_dir.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `outputs_dir.mkdir(parents` untuk dipakai pada langkah berikutnya.
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Menyimpan nilai ke `settings.STORAGE_DIR` untuk dipakai pada langkah berikutnya.
    settings.STORAGE_DIR = storage_dir
    # Menyimpan nilai ke `settings.INPUTS_DIR` untuk dipakai pada langkah berikutnya.
    settings.INPUTS_DIR = inputs_dir
    # Menyimpan nilai ke `settings.OUTPUTS_DIR` untuk dipakai pada langkah berikutnya.
    settings.OUTPUTS_DIR = outputs_dir

    # Menyimpan nilai ke `app` untuk dipakai pada langkah berikutnya.
    app = create_app()
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return TestClient(app)


# Membuat langkah kerja bernama `test_prediction_detail_and_zip_flow`.
def test_prediction_detail_and_zip_flow(tmp_path: Path) -> None:
    # Menyimpan nilai ke `client` untuk dipakai pada langkah berikutnya.
    client = _create_isolated_app(tmp_path)

    # Menyimpan nilai ke `response` untuk dipakai pada langkah berikutnya.
    response = client.post(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "/api/predict",
        # Menyimpan nilai ke `files` untuk dipakai pada langkah berikutnya.
        files=_valid_upload_files(),
        # Menyimpan nilai ke `data` untuk dipakai pada langkah berikutnya.
        data={"threshold": "0.5", "horizon": "2"},
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert response.status_code == 200
    # Menyimpan nilai ke `payload` untuk dipakai pada langkah berikutnya.
    payload = response.json()
    # Menyimpan nilai ke `prediction_id` untuk dipakai pada langkah berikutnya.
    prediction_id = payload["prediction_id"]

    # Menyimpan nilai ke `detail` untuk dipakai pada langkah berikutnya.
    detail = client.get(f"/api/predictions/{prediction_id}")
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert detail.status_code == 200
    # Menyimpan nilai ke `detail_payload` untuk dipakai pada langkah berikutnya.
    detail_payload = detail.json()
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert detail_payload["prediction_id"] == prediction_id
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert detail_payload["download_zip_url"].endswith("/download")
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert "H0" in detail_payload["input_urls"]

    # Menyimpan nilai ke `zip_response` untuk dipakai pada langkah berikutnya.
    zip_response = client.get(f"/api/predictions/{prediction_id}/download")
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert zip_response.status_code == 200
    # Menyimpan nilai ke `archive` untuk dipakai pada langkah berikutnya.
    archive = zipfile.ZipFile(io.BytesIO(zip_response.content))
    # Menyimpan nilai ke `names` untuk dipakai pada langkah berikutnya.
    names = set(archive.namelist())
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert any(name.startswith("inputs/H0.") for name in names)
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert "outputs/params.json" in names
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert any(name.startswith("outputs/heatmap_H+1") for name in names)


# Membuat langkah kerja bernama `test_predict_reject_invalid_file_count`.
def test_predict_reject_invalid_file_count(tmp_path: Path) -> None:
    # Menyimpan nilai ke `client` untuk dipakai pada langkah berikutnya.
    client = _create_isolated_app(tmp_path)
    # Menyimpan nilai ke `short_files` untuk dipakai pada langkah berikutnya.
    short_files = _valid_upload_files()[:6]

    # Menyimpan nilai ke `response` untuk dipakai pada langkah berikutnya.
    response = client.post("/api/predict", files=short_files)
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert response.status_code == 400
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert "7 citra berurutan" in response.json()["detail"]


# Membuat langkah kerja bernama `test_predict_reject_non_image_mime`.
def test_predict_reject_non_image_mime(tmp_path: Path) -> None:
    # Menyimpan nilai ke `client` untuk dipakai pada langkah berikutnya.
    client = _create_isolated_app(tmp_path)
    # Menyimpan nilai ke `files` untuk dipakai pada langkah berikutnya.
    files = _valid_upload_files()
    # Menyimpan nilai ke `files[0]` untuk dipakai pada langkah berikutnya.
    files[0] = ("files", ("H-6.png", b"not-an-image", "text/plain"))

    # Menyimpan nilai ke `response` untuk dipakai pada langkah berikutnya.
    response = client.post("/api/predict", files=files)
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert response.status_code == 400
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert "MIME type" in response.json()["detail"]


# Membuat langkah kerja bernama `test_runtime_status_endpoint`.
def test_runtime_status_endpoint(tmp_path: Path) -> None:
    # Menyimpan nilai ke `client` untuk dipakai pada langkah berikutnya.
    client = _create_isolated_app(tmp_path)
    # Menyimpan nilai ke `response` untuk dipakai pada langkah berikutnya.
    response = client.get("/api/runtime/status")
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert response.status_code == 200
    # Menyimpan nilai ke `payload` untuk dipakai pada langkah berikutnya.
    payload = response.json()
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert "prediction_engine" in payload
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert "model_candidates" in payload
    # Mengecek bahwa hasil test sesuai dengan yang diharapkan.
    assert isinstance(payload["model_candidates"], list)
