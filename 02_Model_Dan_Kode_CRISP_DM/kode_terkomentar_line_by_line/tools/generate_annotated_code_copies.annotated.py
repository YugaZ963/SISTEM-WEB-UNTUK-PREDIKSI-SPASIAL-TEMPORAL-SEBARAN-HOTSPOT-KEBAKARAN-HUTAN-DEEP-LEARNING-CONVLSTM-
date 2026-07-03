# File anotasi dari `tools/generate_annotated_code_copies.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: kode pendukung project skripsi.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Generator dokumentasi kode per baris untuk kebutuhan belajar dan lampiran.

# Isi catatan penjelasan pada bagian kode ini.
Script ini membuat salinan file kode yang diberi komentar di atas setiap baris
# Isi catatan penjelasan pada bagian kode ini.
tanpa mengubah source code aktif. Pendekatan ini sengaja dipilih agar aplikasi
# Isi catatan penjelasan pada bagian kode ini.
tetap aman dijalankan, tetapi pembaca tetap dapat memahami fungsi setiap baris.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Menjaga cara baca kode Python tetap cocok dengan versi yang digunakan.
from __future__ import annotations

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from dataclasses import dataclass
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import os
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import re
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import shutil


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
ROOT = Path(__file__).resolve().parents[1]
# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
OUTPUT_ROOT = ROOT / "docs" / "kode-terkomentar-line-by-line"

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
INCLUDE_SUFFIXES = {
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".py",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".js",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".html",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".css",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".bat",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".sh",
# Menutup susunan data atau perintah yang dimulai sebelumnya.
}

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
EXCLUDED_PARTS = {
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".git",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".pytest_cache",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".venv",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".venv-wsl",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".venv2",
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".venv3",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "site-packages",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "__pycache__",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "artifacts",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "backend_legacy_before_historical_risk_20260416",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "backend/storage",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "docs/kode-terkomentar-line-by-line",
    # Melanjutkan langkah kerja pada bagian kode ini.
    "Laporan Skripsi Yuga - (2) - Copy",
# Menutup susunan data atau perintah yang dimulai sebelumnya.
}

# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
EXCLUDED_FILE_PATTERNS = {
    # Melanjutkan langkah kerja pada bagian kode ini.
    "CUsersYuga Azka Al Razzak",
# Menutup susunan data atau perintah yang dimulai sebelumnya.
}


# Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya.
@dataclass(frozen=True)
# Membuat wadah bernama `LanguageStyle` untuk menyimpan data atau aturan kerja.
class LanguageStyle:
    # Menjelaskan data `name` yang disimpan atau dikirim pada bagian ini.
    name: str
    # Menyimpan nilai ke `line_prefix` untuk dipakai pada langkah berikutnya.
    line_prefix: str | None = None
    # Menyimpan nilai ke `block_start` untuk dipakai pada langkah berikutnya.
    block_start: str | None = None
    # Menyimpan nilai ke `block_prefix` untuk dipakai pada langkah berikutnya.
    block_prefix: str | None = None
    # Menyimpan nilai ke `block_end` untuk dipakai pada langkah berikutnya.
    block_end: str | None = None


# Menentukan nilai tetap yang dipakai sebagai pengaturan proses.
STYLE_BY_SUFFIX = {
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".py": LanguageStyle("python", "#"),
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".js": LanguageStyle("javascript", "//"),
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".sh": LanguageStyle("shell", "#"),
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".bat": LanguageStyle("batch", "REM"),
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".css": LanguageStyle("css", None, "/*", " *", " */"),
    # Melanjutkan langkah kerja pada bagian kode ini.
    ".html": LanguageStyle("html", None, "<!--", "  ", "-->"),
# Menutup susunan data atau perintah yang dimulai sebelumnya.
}


# Membuat langkah kerja bernama `is_excluded`.
def is_excluded(path: Path) -> bool:
    # Menyimpan nilai ke `relative` untuk dipakai pada langkah berikutnya.
    relative = path.relative_to(ROOT)
    # Menyimpan nilai ke `normalized` untuk dipakai pada langkah berikutnya.
    normalized = relative.as_posix()
    # Mengecek syarat sebelum melanjutkan proses.
    if any(pattern in normalized for pattern in EXCLUDED_FILE_PATTERNS):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return True
    # Menyimpan nilai ke `parts` untuk dipakai pada langkah berikutnya.
    parts = set(relative.parts)
    # Mengecek syarat sebelum melanjutkan proses.
    if parts.intersection(EXCLUDED_PARTS):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return True
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return any(normalized.startswith(part + "/") for part in EXCLUDED_PARTS if "/" in part)


# Membuat langkah kerja bernama `discover_code_files`.
def discover_code_files() -> list[Path]:
    # Menyimpan nilai ke `files` untuk dipakai pada langkah berikutnya.
    files: list[Path] = []
    # Mengulang proses untuk setiap data dalam daftar.
    for current_root, dir_names, file_names in os.walk(ROOT):
        # Menyimpan nilai ke `current_path` untuk dipakai pada langkah berikutnya.
        current_path = Path(current_root)
        # Mengecek syarat sebelum melanjutkan proses.
        if is_excluded(current_path) and current_path != ROOT:
            # Menyimpan nilai ke `dir_names[` untuk dipakai pada langkah berikutnya.
            dir_names[:] = []
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue

        # Menyimpan nilai ke `dir_names[` untuk dipakai pada langkah berikutnya.
        dir_names[:] = [
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            name
            # Mengulang proses untuk setiap data dalam daftar.
            for name in dir_names
            # Mengecek syarat sebelum melanjutkan proses.
            if not is_excluded(current_path / name)
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]

        # Mengulang proses untuk setiap data dalam daftar.
        for file_name in file_names:
            # Menyimpan nilai ke `path` untuk dipakai pada langkah berikutnya.
            path = current_path / file_name
            # Mengecek syarat sebelum melanjutkan proses.
            if path.suffix.lower() not in INCLUDE_SUFFIXES:
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Mengecek syarat sebelum melanjutkan proses.
            if is_excluded(path):
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Melanjutkan langkah kerja pada bagian kode ini.
            files.append(path)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return sorted(files, key=lambda item: item.as_posix().lower())


# Membuat langkah kerja bernama `leading_whitespace`.
def leading_whitespace(line: str) -> str:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return line[: len(line) - len(line.lstrip())]


# Membuat langkah kerja bernama `clean_code`.
def clean_code(line: str) -> str:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return line.strip()


# Membuat langkah kerja bernama `explain_python`.
def explain_python(code: str) -> str:
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("#"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Catatan asli dari pembuat kode."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith('"""') or code.startswith("'''"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Catatan pembuka/penutup yang menjelaskan isi bagian kode."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("from __future__ import"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menjaga cara baca kode Python tetap cocok dengan versi yang digunakan."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("import "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengambil alat bantu/library yang diperlukan oleh file ini."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("from ") and " import " in code:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengambil bagian tertentu dari library agar bisa dipakai di file ini."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("@"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("class "):
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = code.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Membuat wadah bernama `{name}` untuk menyimpan data atau aturan kerja."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("def "):
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = code.split("def ", 1)[1].split("(", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Membuat langkah kerja bernama `{name}`."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("async def "):
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = code.split("async def ", 1)[1].split("(", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Membuat langkah kerja `{name}` yang bisa menunggu proses web/backend selesai."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("return "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya."
    # Mengecek syarat sebelum melanjutkan proses.
    if code == "return":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengakhiri langkah kerja tanpa mengirim nilai khusus."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("if "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengecek syarat sebelum melanjutkan proses."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("elif "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengecek syarat lain jika syarat sebelumnya tidak terpenuhi."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("else"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("for "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengulang proses untuk setiap data dalam daftar."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("while "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengulang proses selama syaratnya masih terpenuhi."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("with "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("try"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mencoba menjalankan proses yang mungkin gagal."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("except "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menangani kesalahan agar program tidak langsung berhenti."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("finally"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menjalankan langkah penutup setelah proses percobaan selesai."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("raise "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menghentikan proses dan memberi pesan kesalahan yang jelas."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("assert "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengecek bahwa hasil test sesuai dengan yang diharapkan."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("print("):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menampilkan informasi ke terminal agar proses mudah dicek."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("parser.add_argument"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menambahkan pilihan saat script dijalankan dari terminal."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("np."):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengolah angka piksel, mask hotspot, atau peta probabilitas."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("Image."):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengolah file gambar, seperti membuka atau mengubah citra hotspot."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("rgb = image.convert"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengubah citra ke warna RGB agar format gambar seragam."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("hsv ="):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengubah warna RGB ke HSV agar warna merah hotspot lebih mudah dipisahkan."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("h = hsv"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengambil komponen warna H (hue) untuk mengenali warna merah."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("s = hsv"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengambil komponen S (saturation) untuk memastikan warna cukup kuat."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("v = hsv"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengambil komponen V (brightness) untuk memastikan piksel cukup terang."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("red_low ="):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menandai piksel merah pada rentang warna merah bagian bawah."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("red_high ="):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menandai piksel merah pada rentang warna merah bagian atas."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("mask ="):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menggabungkan hasil deteksi merah menjadi mask hotspot."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("mask_image = Image.fromarray"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengubah mask angka menjadi gambar hitam-putih."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("mask_image = mask_image.filter"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memperbesar titik hotspot kecil agar tidak mudah hilang."
    # Mengecek syarat sebelum melanjutkan proses.
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", code) and "=" not in code:
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = code.split(":", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Menjelaskan data `{name}` yang disimpan atau dikirim pada bagian ini."
    # Mengecek syarat sebelum melanjutkan proses.
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*,?$", code):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menyebutkan item yang ikut dipakai pada daftar di atas."
    # Mengecek syarat sebelum melanjutkan proses.
    if re.match(r"^[A-Z0-9_]+\s*=", code):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menentukan nilai tetap yang dipakai sebagai pengaturan proses."
    # Mengecek syarat sebelum melanjutkan proses.
    if "=" in code and not code.startswith(("==", "!=", ">=", "<=")):
        # Menyimpan nilai ke `left` untuk dipakai pada langkah berikutnya.
        left = code.split("=", 1)[0].strip()
        # Mengecek syarat sebelum melanjutkan proses.
        if left:
            # Mengecek syarat sebelum melanjutkan proses.
            if ":" in left:
                # Menyimpan nilai ke `left` untuk dipakai pada langkah berikutnya.
                left = left.split(":", 1)[0].strip()
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return f"Menyimpan nilai ke `{left}` untuk dipakai pada langkah berikutnya."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith(("]", "}", ")")):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menutup susunan data atau perintah yang dimulai sebelumnya."
    # Mengecek syarat sebelum melanjutkan proses.
    if code in {"[", "{", "("}:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memulai susunan data atau perintah yang ditulis beberapa baris."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.endswith("("):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi."
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "Melanjutkan langkah kerja pada bagian kode ini."


# Membuat langkah kerja bernama `explain_javascript`.
def explain_javascript(code: str) -> str:
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("//"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Catatan asli pada script halaman web."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("const "):
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = code.split("const ", 1)[1].split("=", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Menyimpan data tetap atau elemen halaman ke `{name}`."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("let "):
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = code.split("let ", 1)[1].split("=", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Menyimpan data sementara ke `{name}` yang bisa berubah saat halaman dipakai."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("function "):
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = code.split("function ", 1)[1].split("(", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Membuat langkah kerja `{name}` untuk interaksi halaman."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("async function "):
        # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
        name = code.split("async function ", 1)[1].split("(", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Membuat langkah kerja `{name}` yang bisa menunggu jawaban dari backend."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("if "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengecek syarat sebelum halaman menjalankan proses."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("else"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menjalankan pilihan cadangan jika syarat sebelumnya tidak terpenuhi."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("for "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengulang proses untuk setiap data yang ditampilkan di halaman."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("return "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengirim hasil dari langkah kerja JavaScript."
    # Mengecek syarat sebelum melanjutkan proses.
    if "fetch(" in code:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menghubungi backend untuk mengirim input atau mengambil hasil prediksi."
    # Mengecek syarat sebelum melanjutkan proses.
    if ".addEventListener" in code:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Membuat halaman merespons aksi pengguna, seperti klik atau upload."
    # Mengecek syarat sebelum melanjutkan proses.
    if ".innerHTML" in code:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengganti isi tampilan agar hasil terbaru muncul di halaman."
    # Mengecek syarat sebelum melanjutkan proses.
    if ".textContent" in code:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengubah tulisan yang terlihat pada dashboard."
    # Mengecek syarat sebelum melanjutkan proses.
    if ".classList" in code:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error."
    # Mengecek syarat sebelum melanjutkan proses.
    if "=" in code:
        # Menyimpan nilai ke `left` untuk dipakai pada langkah berikutnya.
        left = code.split("=", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Mengatur nilai `{left}` untuk memperbarui data atau tampilan halaman."
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "Melanjutkan proses interaksi pada dashboard web."


# Membuat langkah kerja bernama `explain_html`.
def explain_html(code: str) -> str:
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<!--"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Catatan HTML untuk menjelaskan bagian halaman."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<!doctype"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menandai file ini sebagai halaman HTML."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<html"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memulai struktur utama halaman web."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<head"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memulai bagian pengaturan halaman yang tidak langsung terlihat."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<body"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memulai bagian halaman yang dilihat pengguna."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<link"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menghubungkan file tampilan/style untuk mempercantik halaman."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<script"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memuat script agar halaman bisa berinteraksi dengan pengguna."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<form"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Membuat area input untuk upload citra dan pengaturan prediksi."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<input"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Membuat kolom input pengguna, seperti file atau threshold."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<button"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Membuat tombol yang bisa diklik pengguna."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<img"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menampilkan gambar, seperti heatmap, overlay, atau preview."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("<table"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Membuat tabel untuk menampilkan riwayat atau ringkasan data."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("</"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menutup bagian halaman yang sudah dibuka sebelumnya."
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "Menyusun tampilan halaman sistem web."


# Membuat langkah kerja bernama `explain_css`.
def explain_css(code: str) -> str:
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("/*"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Catatan CSS untuk menjelaskan pengaturan tampilan."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("@media"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengatur tampilan agar tetap rapi di ukuran layar tertentu."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("@"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menggunakan aturan khusus untuk tampilan halaman."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.endswith("{"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Memulai pengaturan tampilan untuk elemen tertentu."
    # Mengecek syarat sebelum melanjutkan proses.
    if code == "}":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menutup bagian pengaturan tampilan."
    # Mengecek syarat sebelum melanjutkan proses.
    if ":" in code:
        # Menyimpan nilai ke `prop` untuk dipakai pada langkah berikutnya.
        prop = code.split(":", 1)[0].strip()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"Mengatur tampilan `{prop}` pada elemen halaman."
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "Mengatur tampilan visual dashboard."


# Membuat langkah kerja bernama `explain_shell`.
def explain_shell(code: str) -> str:
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("#!"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menentukan program yang menjalankan script ini."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("#"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Catatan pada script terminal."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.startswith("set "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengatur cara script terminal dijalankan."
    # Mengecek syarat sebelum melanjutkan proses.
    if "python" in code.lower():
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menjalankan script Python untuk training, evaluasi, atau dokumentasi."
    # Mengecek syarat sebelum melanjutkan proses.
    if code.endswith("\\"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Melanjutkan perintah ke baris berikutnya agar mudah dibaca."
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "Menjalankan perintah terminal untuk kebutuhan project."


# Membuat langkah kerja bernama `explain_batch`.
def explain_batch(code: str) -> str:
    # Menyimpan nilai ke `lower` untuk dipakai pada langkah berikutnya.
    lower = code.lower()
    # Mengecek syarat sebelum melanjutkan proses.
    if lower.startswith("rem"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Catatan pada script Windows."
    # Mengecek syarat sebelum melanjutkan proses.
    if lower.startswith("@echo"):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengatur apakah perintah ditampilkan saat script berjalan."
    # Mengecek syarat sebelum melanjutkan proses.
    if lower.startswith("set "):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Mengatur nilai sementara yang dibutuhkan proses Windows."
    # Mengecek syarat sebelum melanjutkan proses.
    if "python" in lower:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Menjalankan script Python lewat file Windows."
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "Menjalankan perintah Windows untuk kebutuhan project."


# Membuat langkah kerja bernama `explain_line`.
def explain_line(code: str, style: LanguageStyle) -> str:
    # Mengecek syarat sebelum melanjutkan proses.
    if style.name == "python":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return explain_python(code)
    # Mengecek syarat sebelum melanjutkan proses.
    if style.name == "javascript":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return explain_javascript(code)
    # Mengecek syarat sebelum melanjutkan proses.
    if style.name == "html":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return explain_html(code)
    # Mengecek syarat sebelum melanjutkan proses.
    if style.name == "css":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return explain_css(code)
    # Mengecek syarat sebelum melanjutkan proses.
    if style.name == "shell":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return explain_shell(code)
    # Mengecek syarat sebelum melanjutkan proses.
    if style.name == "batch":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return explain_batch(code)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "Menjalankan baris kode sesuai fungsi file."


# Membuat langkah kerja bernama `comment_line`.
def comment_line(text: str, style: LanguageStyle, indent: str) -> str:
    # Mengecek syarat sebelum melanjutkan proses.
    if style.line_prefix:
        # Menyimpan nilai ke `separator` untuk dipakai pada langkah berikutnya.
        separator = " " if style.line_prefix != "REM" else " "
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return f"{indent}{style.line_prefix}{separator}{text}"
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return f"{indent}{style.block_start} {text} {style.block_end}"


# Membuat langkah kerja bernama `output_path_for`.
def output_path_for(source: Path) -> Path:
    # Menyimpan nilai ke `relative` untuk dipakai pada langkah berikutnya.
    relative = source.relative_to(ROOT)
    # Menyimpan nilai ke `suffix` untuk dipakai pada langkah berikutnya.
    suffix = source.suffix
    # Menyimpan nilai ke `output_name` untuk dipakai pada langkah berikutnya.
    output_name = source.name[: -len(suffix)] + f".annotated{suffix}" if suffix else source.name + ".annotated"
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return OUTPUT_ROOT / relative.parent / output_name


# Membuat langkah kerja bernama `thesis_context_for`.
def thesis_context_for(source: Path) -> str:
    # Menyimpan nilai ke `relative` untuk dipakai pada langkah berikutnya.
    relative = source.relative_to(ROOT).as_posix().lower()
    # Menyimpan nilai ke `name` untuk dipakai pada langkah berikutnya.
    name = source.name.lower()

    # Mengecek syarat sebelum melanjutkan proses.
    if "data_understanding" in name:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Konteks laporan: Data Understanding, yaitu memahami dataset citra hotspot."
    # Mengecek syarat sebelum melanjutkan proses.
    if "data_preparation" in name or "preprocess" in name:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Konteks laporan: Data Preparation, yaitu menyiapkan citra menjadi mask, sequence, dan patch."
    # Mengecek syarat sebelum melanjutkan proses.
    if "modeling" in name or "train" in name or "convlstm" in name or "run_train" in name:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM."
    # Mengecek syarat sebelum melanjutkan proses.
    if "evaluation" in name or "validate" in name or "metric" in name or "geospatial" in name:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Konteks laporan: Evaluation, yaitu mengukur hasil prediksi dan validasi."
    # Mengecek syarat sebelum melanjutkan proses.
    if "backend/" in relative or "frontend/" in relative or "static/" in relative or "deployment" in name:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Konteks laporan: Deployment, yaitu menjalankan model di sistem web."
    # Mengecek syarat sebelum melanjutkan proses.
    if "test" in relative:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return "Konteks laporan: Pengujian sistem, yaitu memastikan fitur berjalan benar."
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return "Konteks laporan: kode pendukung project skripsi."


# Membuat langkah kerja bernama `annotate_file`.
def annotate_file(source: Path) -> Path:
    # Menyimpan nilai ke `style` untuk dipakai pada langkah berikutnya.
    style = STYLE_BY_SUFFIX[source.suffix.lower()]
    # Menyimpan nilai ke `text` untuk dipakai pada langkah berikutnya.
    text = source.read_text(encoding="utf-8", errors="replace")
    # Menyimpan nilai ke `output_lines` untuk dipakai pada langkah berikutnya.
    output_lines: list[str] = []
    # Menyimpan nilai ke `inside_python_docstring` untuk dipakai pada langkah berikutnya.
    inside_python_docstring = False

    # Menyimpan nilai ke `relative` untuk dipakai pada langkah berikutnya.
    relative = source.relative_to(ROOT).as_posix()
    # Melanjutkan langkah kerja pada bagian kode ini.
    output_lines.append(comment_line(f"File anotasi dari `{relative}`.", style, ""))
    # Melanjutkan langkah kerja pada bagian kode ini.
    output_lines.append(comment_line("File ini untuk belajar/lampiran, bukan source utama aplikasi.", style, ""))
    # Melanjutkan langkah kerja pada bagian kode ini.
    output_lines.append(comment_line(thesis_context_for(source), style, ""))
    # Melanjutkan langkah kerja pada bagian kode ini.
    output_lines.append(comment_line("Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.", style, ""))

    # Mengulang proses untuk setiap data dalam daftar.
    for raw_line in text.splitlines():
        # Menyimpan nilai ke `code` untuk dipakai pada langkah berikutnya.
        code = clean_code(raw_line)
        # Mengecek syarat sebelum melanjutkan proses.
        if not code:
            # Melanjutkan langkah kerja pada bagian kode ini.
            output_lines.append("")
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            continue
        # Menyimpan nilai ke `indent` untuk dipakai pada langkah berikutnya.
        indent = leading_whitespace(raw_line)
        # Menyimpan nilai ke `explanation` untuk dipakai pada langkah berikutnya.
        explanation = explain_line(code, style)
        # Mengecek syarat sebelum melanjutkan proses.
        if style.name == "python" and inside_python_docstring and not (
            # Melanjutkan langkah kerja pada bagian kode ini.
            code.startswith('"""') or code.startswith("'''")
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ):
            # Menyimpan nilai ke `explanation` untuk dipakai pada langkah berikutnya.
            explanation = "Isi catatan penjelasan pada bagian kode ini."
        # Melanjutkan langkah kerja pada bagian kode ini.
        output_lines.append(comment_line(explanation, style, indent))
        # Melanjutkan langkah kerja pada bagian kode ini.
        output_lines.append(raw_line)
        # Mengecek syarat sebelum melanjutkan proses.
        if style.name == "python" and (code.startswith('"""') or code.startswith("'''")):
            # Menyimpan nilai ke `quote` untuk dipakai pada langkah berikutnya.
            quote = '"""' if code.startswith('"""') else "'''"
            # Mengecek syarat sebelum melanjutkan proses.
            if code.count(quote) == 1:
                # Menyimpan nilai ke `inside_python_docstring` untuk dipakai pada langkah berikutnya.
                inside_python_docstring = not inside_python_docstring

    # Menyimpan nilai ke `destination` untuk dipakai pada langkah berikutnya.
    destination = output_path_for(source)
    # Menyimpan nilai ke `destination.parent.mkdir(parents` untuk dipakai pada langkah berikutnya.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Menyimpan nilai ke `destination.write_text("\n".join(output_lines) + "\n", encoding` untuk dipakai pada langkah berikutnya.
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return destination


# Membuat langkah kerja bernama `main`.
def main() -> None:
    # Mengecek syarat sebelum melanjutkan proses.
    if OUTPUT_ROOT.exists():
        # Melanjutkan langkah kerja pada bagian kode ini.
        shutil.rmtree(OUTPUT_ROOT)
    # Menyimpan nilai ke `OUTPUT_ROOT.mkdir(parents` untuk dipakai pada langkah berikutnya.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Menyimpan nilai ke `code_files` untuk dipakai pada langkah berikutnya.
    code_files = discover_code_files()
    # Menyimpan nilai ke `outputs` untuk dipakai pada langkah berikutnya.
    outputs = [annotate_file(path) for path in code_files]

    # Menyimpan nilai ke `index_lines` untuk dipakai pada langkah berikutnya.
    index_lines = [
        # Melanjutkan langkah kerja pada bagian kode ini.
        "# Index Kode Terkomentar Line-by-Line",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Folder ini berisi salinan kode teranotasi untuk membantu memahami isi project.",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "File di folder ini bukan source utama dan tidak digunakan oleh aplikasi.",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Komentar dibuat dengan bahasa sederhana agar mudah dipakai untuk belajar, hafalan, dan lampiran skripsi.",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "Label konteks laporan membantu menghubungkan kode dengan CRISP-DM: Data Understanding, Data Preparation, Modeling, Evaluation, dan Deployment.",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
        # Melanjutkan langkah kerja pada bagian kode ini.
        f"Jumlah file teranotasi: {len(outputs)}",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "## Daftar File",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ]
    # Mengulang proses untuk setiap data dalam daftar.
    for output in outputs:
        # Melanjutkan langkah kerja pada bagian kode ini.
        index_lines.append(f"- `{output.relative_to(OUTPUT_ROOT).as_posix()}`")
    # Menyimpan nilai ke `(OUTPUT_ROOT / "README.md").write_text("\n".join(index_lines) + "\n", encoding` untuk dipakai pada langkah berikutnya.
    (OUTPUT_ROOT / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[annotate] File diproses: {len(outputs)}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print(f"[annotate] Output: {OUTPUT_ROOT}")


# Mengecek syarat sebelum melanjutkan proses.
if __name__ == "__main__":
    # Melanjutkan langkah kerja pada bagian kode ini.
    main()
