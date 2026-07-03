"""Komentar file skripsi:
Entry point FastAPI untuk dashboard prediksi area risiko hotspot H+1 berbasis ConvLSTM.

Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
terutama bagian deployment sistem web, integrasi model, dan penyajian hasil prediksi.
"""

# Mengimpor FastAPI sebagai framework utama untuk membangun aplikasi web/API.
from fastapi import FastAPI, Request

# Mengimpor HTMLResponse agar endpoint halaman utama dapat mengembalikan tampilan HTML.
from fastapi.responses import HTMLResponse

# Mengimpor StaticFiles untuk melayani file statis seperti CSS, JavaScript, gambar, dan hasil prediksi.
from fastapi.staticfiles import StaticFiles

# Mengimpor Jinja2Templates untuk merender halaman dashboard berbasis template HTML.
from fastapi.templating import Jinja2Templates

# Mengimpor file konfigurasi backend yang menyimpan path folder, profil model, dan batas upload.
from backend import settings

# Mengimpor router prediksi yang berisi endpoint /api/predict, riwayat, detail, download, dan status runtime.
from backend.routes.predict import router as prediction_router

# Mengimpor service inferensi yang bertugas memuat model ConvLSTM dan menjalankan prediksi.
from backend.services.inference import InferenceService

# Mengimpor ModelSpec sebagai struktur data spesifikasi model aktif yang digunakan sistem.
from backend.services.preprocess import ModelSpec


# Helper ini membuat versi asset dari waktu edit terbaru CSS/JS.
def _static_asset_version() -> str:
    # Daftar file static yang menentukan tampilan dan interaksi dashboard.
    static_files = [
        settings.STATIC_DIR / "styles.css",
        settings.STATIC_DIR / "app.js",
    ]

    # Mencari timestamp terbaru agar URL CSS/JS berubah setelah file static diedit.
    latest_mtime = max((int(path.stat().st_mtime) for path in static_files if path.exists()), default=0)

    # Nilai ini dikirim ke template sebagai query string cache-busting.
    return str(latest_mtime)


# Fungsi factory untuk membuat instance aplikasi FastAPI secara terstruktur.
def create_app() -> FastAPI:
    # Menyiapkan folder kerja seperti storage, inputs, outputs, assets, static, templates, dan models.
    settings.initialize_directories()

    # Membuat object utama FastAPI sebagai aplikasi backend sistem prediksi hotspot.
    app = FastAPI(
        # Menetapkan nama aplikasi yang tampil pada dokumentasi otomatis FastAPI.
        title="Sistem Web Prediksi Hotspot",
        # Menetapkan versi aplikasi untuk menunjukkan perkembangan implementasi sistem.
        version="1.1.0",
        # Menjelaskan fungsi utama aplikasi, yaitu upload 7 citra historis untuk prediksi H+1.
        description="Upload 7 citra berurutan H-6 sampai H0 untuk prediksi area risiko hotspot berbasis ConvLSTM.",
    )

    # Middleware lokal untuk mencegah browser memakai HTML/CSS/JS lama saat server diganti antar-folder.
    @app.middleware("http")
    async def add_local_no_cache_headers(request: Request, call_next):
        # Menjalankan request terlebih dahulu agar response asli dari route/static tetap terbentuk.
        response = await call_next(request)

        # Cache dimatikan hanya untuk halaman utama dan static dashboard, bukan file hasil prediksi di /storage.
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            # Cache-Control no-store memaksa browser mengambil ulang tampilan dan script dari folder server aktif.
            response.headers["Cache-Control"] = "no-store, no-cache, max-age=0, must-revalidate"
            # Header lama ini membantu browser/proxy yang masih membaca aturan HTTP/1.0.
            response.headers["Pragma"] = "no-cache"
            # Expires 0 menandai response sebagai sudah kedaluwarsa.
            response.headers["Expires"] = "0"

        # Mengembalikan response yang sudah diberi aturan cache lokal.
        return response

    # Mengambil konfigurasi model aktif dari settings berdasarkan profile yang sedang digunakan.
    model_config = settings.ACTIVE_MODEL_CONFIG

    # Membentuk spesifikasi model agar parameter model dapat dipakai konsisten oleh service lain.
    model_spec = ModelSpec(
        # Menentukan ukuran grid/input model, misalnya 160 untuk model historical_risk_patch_160.
        grid_size=int(model_config["grid_size"]),
        # Menentukan jumlah channel input model, pada sistem ini umumnya 1 channel mask hotspot.
        channels=int(model_config["channels"]),
        # Menentukan jumlah frame waktu input, yaitu 7 citra dari H-6 sampai H0.
        time_steps=int(model_config["time_steps"]),
        # Menyimpan nama profile model aktif, misalnya historical_risk_patch_160.
        profile_name=settings.ACTIVE_MODEL_PROFILE,
        # Menyimpan nama tampilan model untuk ditampilkan pada dashboard web.
        display_name=str(model_config["display_name"]),
        # Menentukan mode preprocessing, misalnya hotspot_red_mask untuk ekstraksi area hotspot merah.
        preprocess_mode=str(model_config["preprocess_mode"]),
        # Menentukan mode inferensi, misalnya patch_stitch untuk prediksi berbasis potongan patch.
        inference_mode=str(model_config["inference_mode"]),
        # Menentukan ukuran patch input model jika mode inferensi menggunakan patch_stitch.
        patch_size=model_config["patch_size"],
        # Menentukan jarak pergeseran patch agar hasil prediksi dapat digabung kembali.
        patch_stride=model_config["patch_stride"],
        # Menentukan jumlah patch yang diproses dalam satu batch saat inferensi model.
        patch_batch_size=model_config["patch_batch_size"],
        # Menentukan ukuran kernel dilasi untuk memperjelas mask hotspot pada tahap preprocessing.
        input_dilation_kernel=int(model_config["input_dilation_kernel"]),
        # Menyimpan threshold rekomendasi untuk pembuatan binary mask risiko hotspot.
        recommended_threshold=model_config["recommended_threshold"],
    )

    # Membuat service inferensi berdasarkan spesifikasi model aktif.
    inference_service = InferenceService(spec=model_spec)

    # Memuat model ConvLSTM jika tersedia; jika gagal, service akan memakai fallback heuristik.
    inference_service.load_model()

    # Menyimpan spesifikasi model pada app.state agar dapat diakses oleh seluruh endpoint API.
    app.state.model_spec = model_spec

    # Menyimpan service inferensi pada app.state agar endpoint prediksi dapat menjalankan model.
    app.state.inference_service = inference_service

    # Menyiapkan folder template HTML yang dipakai untuk menampilkan dashboard utama.
    templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

    # Menyimpan object template ke app.state agar tetap tersedia selama aplikasi berjalan.
    app.state.templates = templates

    # Melayani file statis frontend seperti styles.css dan app.js melalui URL /static.
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

    # Melayani file hasil input/output prediksi melalui URL /storage.
    app.mount("/storage", StaticFiles(directory=str(settings.STORAGE_DIR)), name="storage")

    # Mendaftarkan seluruh endpoint API prediksi dari backend/routes/predict.py ke aplikasi utama.
    app.include_router(prediction_router)

    # Mendaftarkan endpoint halaman utama dashboard ketika user membuka URL root (/).
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        # Merender template index.html dan mengirim konfigurasi penting ke frontend dashboard.
        return templates.TemplateResponse(
            # Menentukan file template dashboard yang akan ditampilkan ke pengguna.
            "index.html",
            # Mengirim data backend ke template agar dashboard sesuai konfigurasi model aktif.
            {
                # Request wajib dikirim ke Jinja2 agar template FastAPI dapat dirender dengan benar.
                "request": request,
                # Mengirim daftar nama file wajib, yaitu H-6 sampai H0.
                "required_stems": settings.REQUIRED_STEMS,
                # Mengirim batas ukuran file maksimum dalam MB untuk ditampilkan pada form upload.
                "max_file_mb": settings.MAX_FILE_MB,
                # Mengirim ukuran grid model aktif untuk ditampilkan pada dashboard.
                "grid_size": model_spec.grid_size,
                # Mengirim nama profile model aktif untuk informasi runtime sistem.
                "model_profile": model_spec.profile_name,
                # Mengirim nama tampilan model aktif agar user tahu model yang sedang digunakan.
                "model_display_name": model_spec.display_name,
                # Mengirim mode inferensi model, misalnya patch_stitch atau direct_resize.
                "inference_mode": model_spec.inference_mode,
                # Mengirim threshold rekomendasi untuk membantu user menentukan batas risiko.
                "recommended_threshold": model_spec.recommended_threshold,
                # Mengirim versi CSS/JS agar browser tidak memakai cache dari folder server sebelumnya.
                "static_asset_version": _static_asset_version(),
            },
        )

    # Mengembalikan instance aplikasi FastAPI yang sudah lengkap dengan konfigurasi, route, dan static file.
    return app


# Membuat object app yang akan dijalankan oleh Uvicorn melalui perintah backend.main:app.
app = create_app()
