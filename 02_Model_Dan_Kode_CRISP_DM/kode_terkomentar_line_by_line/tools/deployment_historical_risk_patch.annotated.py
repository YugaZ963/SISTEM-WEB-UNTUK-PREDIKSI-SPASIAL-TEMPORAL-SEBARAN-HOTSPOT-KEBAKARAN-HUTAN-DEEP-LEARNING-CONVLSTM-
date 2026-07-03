# File anotasi dari `tools/deployment_historical_risk_patch.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Script ringkas fase Deployment CRISP-DM untuk menjelaskan integrasi model ke sistem web.

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
import argparse
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import json
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import sys
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Mengecek syarat sebelum melanjutkan proses.
if str(PROJECT_ROOT) not in sys.path:
    # Melanjutkan langkah kerja pada bagian kode ini.
    sys.path.insert(0, str(PROJECT_ROOT))

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend import settings
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.main import create_app


# Membuat langkah kerja bernama `build_parser`.
def build_parser() -> argparse.ArgumentParser:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description=(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Ringkasan fase Deployment historical risk patch. "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Script ini fokus pada wiring model ke backend FastAPI, runtime aktif, dan endpoint utama."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--show-routes",
        # Menyimpan nilai ke `action` untuk dipakai pada langkah berikutnya.
        action="store_true",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Tampilkan daftar route API dan halaman utama.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--json", action="store_true")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser


# Membuat langkah kerja bernama `collect_routes`.
def collect_routes(app) -> list[dict]:
    # Menyimpan nilai ke `routes` untuk dipakai pada langkah berikutnya.
    routes: list[dict] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for route in app.routes:
        # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
        path = getattr(route, "path", None)
        # Menyimpan nilai ke `methods` untuk dipakai pada langkah berikutnya.
        methods = sorted(method for method in (getattr(route, "methods", set()) or set()) if method != "HEAD")
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = getattr(route, "name", None)
        # Mengecek syarat sebelum melanjutkan proses.
        if not path:
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        routes.append(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                # Melanjutkan langkah kerja pada bagian kode ini.
                "path": path,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "methods": methods,
                # Melanjutkan langkah kerja pada bagian kode ini.
                "name": name,
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            }
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return routes


# Membuat langkah kerja bernama `build_summary`.
def build_summary(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `app` untuk dipakai pada langkah berikutnya.
    app = create_app()
    # Menyimpan nilai ke `model_spec` untuk dipakai pada langkah berikutnya.
    model_spec = app.state.model_spec
    # Menyimpan nilai ke `inference_service` untuk dipakai pada langkah berikutnya.
    inference_service = app.state.inference_service
    # Menyimpan nilai ke `routes` untuk dipakai pada langkah berikutnya.
    routes = collect_routes(app)

    # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
    summary = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "phase": "deployment",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "framework": "FastAPI",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "server_entrypoint": "backend.main:app",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "server_run_command": "python -m uvicorn backend.main:app --reload",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "default_url": "http://127.0.0.1:8000",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "active_model_profile": model_spec.profile_name,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "active_model_display_name": model_spec.display_name,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "preprocess_mode": model_spec.preprocess_mode,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "inference_mode": model_spec.inference_mode,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "time_steps": model_spec.time_steps,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "grid_size": model_spec.grid_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "channels": model_spec.channels,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_size": model_spec.patch_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_stride": model_spec.patch_stride,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_batch_size": model_spec.patch_batch_size,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "input_dilation_kernel": model_spec.input_dilation_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "recommended_threshold": model_spec.recommended_threshold,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "prediction_engine": inference_service.backend,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_backend": inference_service.backend,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_path": inference_service.model_path,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "runtime_model_loaded": inference_service.model is not None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "storage_dirs": {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "inputs": str(settings.INPUTS_DIR),
            # Melanjutkan langkah kerja pada bagian kode ini.
            "outputs": str(settings.OUTPUTS_DIR),
            # Melanjutkan langkah kerja pada bagian kode ini.
            "templates": str(settings.TEMPLATES_DIR),
            # Melanjutkan langkah kerja pada bagian kode ini.
            "static": str(settings.STATIC_DIR),
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_candidates": inference_service.list_model_candidates(),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "key_routes": routes,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return summary


# Membuat langkah kerja bernama `print_human_summary`.
def print_human_summary(summary: dict, show_routes: bool) -> None:
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Framework:", summary["framework"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Entrypoint:", summary["server_entrypoint"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Command run:", summary["server_run_command"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] URL default:", summary["default_url"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Model profile aktif:", summary["active_model_profile"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Nama model:", summary["active_model_display_name"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Preprocess mode:", summary["preprocess_mode"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Inference mode:", summary["inference_mode"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Input:", f"{summary['time_steps']} frame | {summary['grid_size']}x{summary['grid_size']} | {summary['channels']} channel")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Patch config:", f"size={summary['patch_size']} stride={summary['patch_stride']} batch={summary['patch_batch_size']}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Threshold rekomendasi:", summary["recommended_threshold"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Prediction engine:", summary["prediction_engine"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Model path:", summary["model_path"] or "-")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Runtime model loaded:", summary["runtime_model_loaded"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Direktori storage:")
    # Mengulang proses untuk setiap data dalam daftar.
    for key, value in summary["storage_dirs"].items():
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"- {key}: {value}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[deployment] Jumlah model candidate:", len(summary["model_candidates"]))

    # Mengecek syarat sebelum melanjutkan proses.
    if show_routes:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[deployment] Routes:")
        # Mengulang proses untuk setiap data dalam daftar.
        for route in summary["key_routes"]:
            # Menyimpan nilai ke `methods` untuk dipakai pada langkah berikutnya.
            methods = ",".join(route["methods"]) if route["methods"] else "-"
            # Menampilkan informasi ke terminal agar proses mudah dicek.
            print(f"- {methods:12s} {route['path']} | name={route['name']}")


# Membuat langkah kerja bernama `main`.
def main() -> None:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = build_parser()
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parser.parse_args()
    # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
    summary = build_summary(args)
    # Menyimpan nilai ke `print_human_summary(summary, show_routes` untuk dipakai pada langkah berikutnya.
    print_human_summary(summary, show_routes=args.show_routes)
    # Mengecek syarat sebelum melanjutkan proses.
    if args.json:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print()
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(json.dumps(summary, indent=2, ensure_ascii=False))


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Melanjutkan langkah kerja pada bagian kode ini.
    main()
