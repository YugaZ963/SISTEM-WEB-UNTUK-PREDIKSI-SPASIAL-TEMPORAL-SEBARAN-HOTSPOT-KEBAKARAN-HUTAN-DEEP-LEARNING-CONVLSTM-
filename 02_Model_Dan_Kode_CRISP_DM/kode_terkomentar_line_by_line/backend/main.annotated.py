# File anotasi dari `backend/main.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Entry point FastAPI untuk dashboard prediksi area risiko hotspot H+1 berbasis ConvLSTM.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi import FastAPI, Request
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi.responses import HTMLResponse
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi.staticfiles import StaticFiles
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from fastapi.templating import Jinja2Templates

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend import settings
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.routes.predict import router as prediction_router
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.services.inference import InferenceService
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.services.preprocess import ModelSpec


# Membuat langkah kerja bernama `create_app`.
def create_app() -> FastAPI:
    # Melanjutkan langkah kerja pada bagian kode ini.
    settings.initialize_directories()

    # Menyimpan nilai ke `app` untuk dipakai pada langkah berikutnya.
    app = FastAPI(
        # Menyimpan nilai ke `title` untuk dipakai pada langkah berikutnya.
        title="Sistem Web Prediksi Hotspot",
        # Menyimpan nilai ke `version` untuk dipakai pada langkah berikutnya.
        version="1.1.0",
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description="Upload 7 citra berurutan H-6 sampai H0 untuk prediksi area risiko hotspot berbasis ConvLSTM.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )

    # Menyimpan nilai ke `model_config` untuk dipakai pada langkah berikutnya.
    model_config = settings.ACTIVE_MODEL_CONFIG
    # Menyimpan nilai ke `model_spec` untuk dipakai pada langkah berikutnya.
    model_spec = ModelSpec(
        # Menyimpan nilai ke `grid_size` untuk dipakai pada langkah berikutnya.
        grid_size=int(model_config["grid_size"]),
        # Menyimpan nilai ke `channels` untuk dipakai pada langkah berikutnya.
        channels=int(model_config["channels"]),
        # Menyimpan nilai ke `time_steps` untuk dipakai pada langkah berikutnya.
        time_steps=int(model_config["time_steps"]),
        # Menyimpan nilai ke `profile_name` untuk dipakai pada langkah berikutnya.
        profile_name=settings.ACTIVE_MODEL_PROFILE,
        # Menyimpan nilai ke `display_name` untuk dipakai pada langkah berikutnya.
        display_name=str(model_config["display_name"]),
        # Menyimpan nilai ke `preprocess_mode` untuk dipakai pada langkah berikutnya.
        preprocess_mode=str(model_config["preprocess_mode"]),
        # Menyimpan nilai ke `inference_mode` untuk dipakai pada langkah berikutnya.
        inference_mode=str(model_config["inference_mode"]),
        # Menyimpan nilai ke `patch_size` untuk dipakai pada langkah berikutnya.
        patch_size=model_config["patch_size"],
        # Menyimpan nilai ke `patch_stride` untuk dipakai pada langkah berikutnya.
        patch_stride=model_config["patch_stride"],
        # Menyimpan nilai ke `patch_batch_size` untuk dipakai pada langkah berikutnya.
        patch_batch_size=model_config["patch_batch_size"],
        # Menyimpan nilai ke `input_dilation_kernel` untuk dipakai pada langkah berikutnya.
        input_dilation_kernel=int(model_config["input_dilation_kernel"]),
        # Menyimpan nilai ke `recommended_threshold` untuk dipakai pada langkah berikutnya.
        recommended_threshold=model_config["recommended_threshold"],
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `inference_service` untuk dipakai pada langkah berikutnya.
    inference_service = InferenceService(spec=model_spec)
    # Melanjutkan langkah kerja pada bagian kode ini.
    inference_service.load_model()

    # Menyimpan nilai ke `app.state.model_spec` untuk dipakai pada langkah berikutnya.
    app.state.model_spec = model_spec
    # Menyimpan nilai ke `app.state.inference_service` untuk dipakai pada langkah berikutnya.
    app.state.inference_service = inference_service

    # Menyimpan nilai ke `templates` untuk dipakai pada langkah berikutnya.
    templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))
    # Menyimpan nilai ke `app.state.templates` untuk dipakai pada langkah berikutnya.
    app.state.templates = templates

    # Menyimpan nilai ke `app.mount("/static", StaticFiles(directory` untuk dipakai pada langkah berikutnya.
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
    # Menyimpan nilai ke `app.mount("/storage", StaticFiles(directory` untuk dipakai pada langkah berikutnya.
    app.mount("/storage", StaticFiles(directory=str(settings.STORAGE_DIR)), name="storage")

    # Melanjutkan langkah kerja pada bagian kode ini.
    app.include_router(prediction_router)

    # Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
    @app.get("/", response_class=HTMLResponse)
    # Membuat langkah kerja `index` yang bisa menunggu proses web/backend selesai.
    async def index(request: Request) -> HTMLResponse:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return templates.TemplateResponse(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "index.html",
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                # Melanjutkan langkah kerja pada bagian kode ini.
                "request": request,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "required_stems": settings.REQUIRED_STEMS,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "max_file_mb": settings.MAX_FILE_MB,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "grid_size": model_spec.grid_size,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "model_profile": model_spec.profile_name,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "model_display_name": model_spec.display_name,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "inference_mode": model_spec.inference_mode,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "recommended_threshold": model_spec.recommended_threshold,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            },
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return app


# Menyimpan nilai ke `app` untuk dipakai pada langkah berikutnya.
app = create_app()
