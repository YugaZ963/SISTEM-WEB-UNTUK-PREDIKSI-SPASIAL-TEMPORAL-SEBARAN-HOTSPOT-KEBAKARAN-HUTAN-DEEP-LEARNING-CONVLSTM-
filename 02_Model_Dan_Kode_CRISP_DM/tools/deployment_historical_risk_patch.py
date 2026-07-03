
"""Komentar file skripsi:
Script ringkas fase Deployment CRISP-DM untuk menjelaskan integrasi model ke sistem web.

Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
terutama data understanding, data preparation, modeling, evaluation,
atau deployment sesuai fungsi teknisnya.
"""

from __future__ import annotations

# argparse dipakai agar parameter dataset, patch, threshold, dan output bisa diatur dari command line.
import argparse
# json dipakai untuk menyimpan ringkasan eksperimen, metrik, dan metadata proses.
import json
# sys dipakai untuk keluar dari script atau mengatur jalur eksekusi command line.
import sys
# Path dipakai agar lokasi dataset, model, output, dan file gambar tetap rapi lintas OS.
from pathlib import Path

# Root project dipakai sebagai dasar path dataset, artefak, dan script pendukung.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import settings
from backend.main import create_app


# Menyusun opsi command line agar fase CRISP-DM bisa dijalankan ulang dengan parameter yang jelas.
def build_parser() -> argparse.ArgumentParser:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description=(
            "Ringkasan fase Deployment historical risk patch. "
            "Script ini fokus pada wiring model ke backend FastAPI, runtime aktif, dan endpoint utama."
        )
    )
    parser.add_argument(
        "--show-routes",
        action="store_true",
        help="Tampilkan daftar route API dan halaman utama.",
    )
    # Opsi ini membuat output ringkasan dicetak sebagai JSON agar mudah dikutip/dibaca ulang.
    parser.add_argument("--json", action="store_true")
    # Hasil ini dikembalikan sebagai output fungsi `build_parser` untuk tahap berikutnya.
    return parser


# Fungsi `collect_routes` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def collect_routes(app) -> list[dict]:
    routes: list[dict] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = sorted(method for method in (getattr(route, "methods", set()) or set()) if method != "HEAD")
        name = getattr(route, "name", None)
        if not path:
            continue
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        routes.append(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            {
                "path": path,
                "methods": methods,
                "name": name,
            }
        )
    # Hasil ini dikembalikan sebagai output fungsi `collect_routes` untuk tahap berikutnya.
    return routes


# Menggabungkan hasil proses menjadi ringkasan JSON/console untuk laporan dan verifikasi eksperimen.
def build_summary(args: argparse.Namespace) -> dict:
    app = create_app()
    model_spec = app.state.model_spec
    inference_service = app.state.inference_service
    routes = collect_routes(app)

    # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
    summary = {
        "phase": "deployment",
        "framework": "FastAPI",
        "server_entrypoint": "backend.main:app",
        "server_run_command": "python -m uvicorn backend.main:app --reload",
        "default_url": "http://127.0.0.1:8000",
        "active_model_profile": model_spec.profile_name,
        "active_model_display_name": model_spec.display_name,
        "preprocess_mode": model_spec.preprocess_mode,
        "inference_mode": model_spec.inference_mode,
        "time_steps": model_spec.time_steps,
        "grid_size": model_spec.grid_size,
        "channels": model_spec.channels,
        "patch_size": model_spec.patch_size,
        "patch_stride": model_spec.patch_stride,
        "patch_batch_size": model_spec.patch_batch_size,
        "input_dilation_kernel": model_spec.input_dilation_kernel,
        "recommended_threshold": model_spec.recommended_threshold,
        "prediction_engine": inference_service.backend,
        "model_backend": inference_service.backend,
        "model_path": inference_service.model_path,
        "runtime_model_loaded": inference_service.model is not None,
        "storage_dirs": {
            "inputs": str(settings.INPUTS_DIR),
            "outputs": str(settings.OUTPUTS_DIR),
            "templates": str(settings.TEMPLATES_DIR),
            "static": str(settings.STATIC_DIR),
        },
        "model_candidates": inference_service.list_model_candidates(),
        "key_routes": routes,
    }
    # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
    return summary


# Fungsi `print_human_summary` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def print_human_summary(summary: dict, show_routes: bool) -> None:
    print("[deployment] Framework:", summary["framework"])
    print("[deployment] Entrypoint:", summary["server_entrypoint"])
    print("[deployment] Command run:", summary["server_run_command"])
    print("[deployment] URL default:", summary["default_url"])
    print("[deployment] Model profile aktif:", summary["active_model_profile"])
    print("[deployment] Nama model:", summary["active_model_display_name"])
    print("[deployment] Preprocess mode:", summary["preprocess_mode"])
    print("[deployment] Inference mode:", summary["inference_mode"])
    print("[deployment] Input:", f"{summary['time_steps']} frame | {summary['grid_size']}x{summary['grid_size']} | {summary['channels']} channel")
    print("[deployment] Patch config:", f"size={summary['patch_size']} stride={summary['patch_stride']} batch={summary['patch_batch_size']}")
    print("[deployment] Threshold rekomendasi:", summary["recommended_threshold"])
    print("[deployment] Prediction engine:", summary["prediction_engine"])
    print("[deployment] Model path:", summary["model_path"] or "-")
    print("[deployment] Runtime model loaded:", summary["runtime_model_loaded"])
    print("[deployment] Direktori storage:")
    for key, value in summary["storage_dirs"].items():
        print(f"- {key}: {value}")
    print("[deployment] Jumlah model candidate:", len(summary["model_candidates"]))

    if show_routes:
        print("[deployment] Routes:")
        for route in summary["key_routes"]:
            methods = ",".join(route["methods"]) if route["methods"] else "-"
            print(f"- {methods:12s} {route['path']} | name={route['name']}")


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> None:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = build_parser()
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parser.parse_args()
    # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
    summary = build_summary(args)
    print_human_summary(summary, show_routes=args.show_routes)
    if args.json:
        print()
        # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
