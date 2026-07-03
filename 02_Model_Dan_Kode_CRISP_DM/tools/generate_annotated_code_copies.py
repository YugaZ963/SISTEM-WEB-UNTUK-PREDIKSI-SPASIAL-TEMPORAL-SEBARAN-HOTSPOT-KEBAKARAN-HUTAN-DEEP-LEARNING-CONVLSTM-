"""Komentar file skripsi:
Generator dokumentasi kode per baris untuk kebutuhan belajar dan lampiran.

Script ini membuat salinan file kode yang diberi komentar di atas setiap baris
tanpa mengubah source code aktif. Pendekatan ini sengaja dipilih agar aplikasi
tetap aman dijalankan, tetapi pembaca tetap dapat memahami fungsi setiap baris.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs" / "kode-terkomentar-line-by-line"

INCLUDE_SUFFIXES = {
    ".py",
    ".js",
    ".html",
    ".css",
    ".bat",
    ".sh",
}

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    ".venv-wsl",
    ".venv2",
    ".venv3",
    "site-packages",
    "__pycache__",
    "artifacts",
    "backend_legacy_before_historical_risk_20260416",
    "backend/storage",
    "docs/kode-terkomentar-line-by-line",
    "Laporan Skripsi Yuga - (2) - Copy",
}

EXCLUDED_FILE_PATTERNS = {
    "CUsersYuga Azka Al Razzak",
}


@dataclass(frozen=True)
class LanguageStyle:
    name: str
    line_prefix: str | None = None
    block_start: str | None = None
    block_prefix: str | None = None
    block_end: str | None = None


STYLE_BY_SUFFIX = {
    ".py": LanguageStyle("python", "#"),
    ".js": LanguageStyle("javascript", "//"),
    ".sh": LanguageStyle("shell", "#"),
    ".bat": LanguageStyle("batch", "REM"),
    ".css": LanguageStyle("css", None, "/*", " *", " */"),
    ".html": LanguageStyle("html", None, "<!--", "  ", "-->"),
}


def is_excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    normalized = relative.as_posix()
    if any(pattern in normalized for pattern in EXCLUDED_FILE_PATTERNS):
        return True
    parts = set(relative.parts)
    if parts.intersection(EXCLUDED_PARTS):
        return True
    return any(normalized.startswith(part + "/") for part in EXCLUDED_PARTS if "/" in part)


def discover_code_files() -> list[Path]:
    files: list[Path] = []
    for current_root, dir_names, file_names in os.walk(ROOT):
        current_path = Path(current_root)
        if is_excluded(current_path) and current_path != ROOT:
            dir_names[:] = []
            continue

        dir_names[:] = [
            name
            for name in dir_names
            if not is_excluded(current_path / name)
        ]

        for file_name in file_names:
            path = current_path / file_name
            if path.suffix.lower() not in INCLUDE_SUFFIXES:
                continue
            if is_excluded(path):
                continue
            files.append(path)
    return sorted(files, key=lambda item: item.as_posix().lower())


def leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def clean_code(line: str) -> str:
    return line.strip()


def explain_python(code: str) -> str:
    if code.startswith("#"):
        return "Catatan asli dari pembuat kode."
    if code.startswith('"""') or code.startswith("'''"):
        return "Catatan pembuka/penutup yang menjelaskan isi bagian kode."
    if code.startswith("from __future__ import"):
        return "Menjaga cara baca kode Python tetap cocok dengan versi yang digunakan."
    if code.startswith("import "):
        return "Mengambil alat bantu/library yang diperlukan oleh file ini."
    if code.startswith("from ") and " import " in code:
        return "Mengambil bagian tertentu dari library agar bisa dipakai di file ini."
    if code.startswith("@"):
        return "Memberi tanda khusus untuk cara kerja fungsi atau class di bawahnya."
    if code.startswith("class "):
        name = code.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0].strip()
        return f"Membuat wadah bernama `{name}` untuk menyimpan data atau aturan kerja."
    if code.startswith("def "):
        name = code.split("def ", 1)[1].split("(", 1)[0].strip()
        return f"Membuat langkah kerja bernama `{name}`."
    if code.startswith("async def "):
        name = code.split("async def ", 1)[1].split("(", 1)[0].strip()
        return f"Membuat langkah kerja `{name}` yang bisa menunggu proses web/backend selesai."
    if code.startswith("return "):
        return "Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya."
    if code == "return":
        return "Mengakhiri langkah kerja tanpa mengirim nilai khusus."
    if code.startswith("if "):
        return "Mengecek syarat sebelum melanjutkan proses."
    if code.startswith("elif "):
        return "Mengecek syarat lain jika syarat sebelumnya tidak terpenuhi."
    if code.startswith("else"):
        return "Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi."
    if code.startswith("for "):
        return "Mengulang proses untuk setiap data dalam daftar."
    if code.startswith("while "):
        return "Mengulang proses selama syaratnya masih terpenuhi."
    if code.startswith("with "):
        return "Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai."
    if code.startswith("try"):
        return "Mencoba menjalankan proses yang mungkin gagal."
    if code.startswith("except "):
        return "Menangani kesalahan agar program tidak langsung berhenti."
    if code.startswith("finally"):
        return "Menjalankan langkah penutup setelah proses percobaan selesai."
    if code.startswith("raise "):
        return "Menghentikan proses dan memberi pesan kesalahan yang jelas."
    if code.startswith("assert "):
        return "Mengecek bahwa hasil test sesuai dengan yang diharapkan."
    if code.startswith("print("):
        return "Menampilkan informasi ke terminal agar proses mudah dicek."
    if code.startswith("parser.add_argument"):
        return "Menambahkan pilihan saat script dijalankan dari terminal."
    if code.startswith("np."):
        return "Mengolah angka piksel, mask hotspot, atau peta probabilitas."
    if code.startswith("Image."):
        return "Mengolah file gambar, seperti membuka atau mengubah citra hotspot."
    if code.startswith("rgb = image.convert"):
        return "Mengubah citra ke warna RGB agar format gambar seragam."
    if code.startswith("hsv ="):
        return "Mengubah warna RGB ke HSV agar warna merah hotspot lebih mudah dipisahkan."
    if code.startswith("h = hsv"):
        return "Mengambil komponen warna H (hue) untuk mengenali warna merah."
    if code.startswith("s = hsv"):
        return "Mengambil komponen S (saturation) untuk memastikan warna cukup kuat."
    if code.startswith("v = hsv"):
        return "Mengambil komponen V (brightness) untuk memastikan piksel cukup terang."
    if code.startswith("red_low ="):
        return "Menandai piksel merah pada rentang warna merah bagian bawah."
    if code.startswith("red_high ="):
        return "Menandai piksel merah pada rentang warna merah bagian atas."
    if code.startswith("mask ="):
        return "Menggabungkan hasil deteksi merah menjadi mask hotspot."
    if code.startswith("mask_image = Image.fromarray"):
        return "Mengubah mask angka menjadi gambar hitam-putih."
    if code.startswith("mask_image = mask_image.filter"):
        return "Memperbesar titik hotspot kecil agar tidak mudah hilang."
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", code) and "=" not in code:
        name = code.split(":", 1)[0].strip()
        return f"Menjelaskan data `{name}` yang disimpan atau dikirim pada bagian ini."
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*,?$", code):
        return "Menyebutkan item yang ikut dipakai pada daftar di atas."
    if re.match(r"^[A-Z0-9_]+\s*=", code):
        return "Menentukan nilai tetap yang dipakai sebagai pengaturan proses."
    if "=" in code and not code.startswith(("==", "!=", ">=", "<=")):
        left = code.split("=", 1)[0].strip()
        if left:
            if ":" in left:
                left = left.split(":", 1)[0].strip()
            return f"Menyimpan nilai ke `{left}` untuk dipakai pada langkah berikutnya."
    if code.startswith(("]", "}", ")")):
        return "Menutup susunan data atau perintah yang dimulai sebelumnya."
    if code in {"[", "{", "("}:
        return "Memulai susunan data atau perintah yang ditulis beberapa baris."
    if code.endswith("("):
        return "Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi."
    return "Melanjutkan langkah kerja pada bagian kode ini."


def explain_javascript(code: str) -> str:
    if code.startswith("//"):
        return "Catatan asli pada script halaman web."
    if code.startswith("const "):
        name = code.split("const ", 1)[1].split("=", 1)[0].strip()
        return f"Menyimpan data tetap atau elemen halaman ke `{name}`."
    if code.startswith("let "):
        name = code.split("let ", 1)[1].split("=", 1)[0].strip()
        return f"Menyimpan data sementara ke `{name}` yang bisa berubah saat halaman dipakai."
    if code.startswith("function "):
        name = code.split("function ", 1)[1].split("(", 1)[0].strip()
        return f"Membuat langkah kerja `{name}` untuk interaksi halaman."
    if code.startswith("async function "):
        name = code.split("async function ", 1)[1].split("(", 1)[0].strip()
        return f"Membuat langkah kerja `{name}` yang bisa menunggu jawaban dari backend."
    if code.startswith("if "):
        return "Mengecek syarat sebelum halaman menjalankan proses."
    if code.startswith("else"):
        return "Menjalankan pilihan cadangan jika syarat sebelumnya tidak terpenuhi."
    if code.startswith("for "):
        return "Mengulang proses untuk setiap data yang ditampilkan di halaman."
    if code.startswith("return "):
        return "Mengirim hasil dari langkah kerja JavaScript."
    if "fetch(" in code:
        return "Menghubungi backend untuk mengirim input atau mengambil hasil prediksi."
    if ".addEventListener" in code:
        return "Membuat halaman merespons aksi pengguna, seperti klik atau upload."
    if ".innerHTML" in code:
        return "Mengganti isi tampilan agar hasil terbaru muncul di halaman."
    if ".textContent" in code:
        return "Mengubah tulisan yang terlihat pada dashboard."
    if ".classList" in code:
        return "Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error."
    if "=" in code:
        left = code.split("=", 1)[0].strip()
        return f"Mengatur nilai `{left}` untuk memperbarui data atau tampilan halaman."
    return "Melanjutkan proses interaksi pada dashboard web."


def explain_html(code: str) -> str:
    if code.startswith("<!--"):
        return "Catatan HTML untuk menjelaskan bagian halaman."
    if code.startswith("<!doctype"):
        return "Menandai file ini sebagai halaman HTML."
    if code.startswith("<html"):
        return "Memulai struktur utama halaman web."
    if code.startswith("<head"):
        return "Memulai bagian pengaturan halaman yang tidak langsung terlihat."
    if code.startswith("<body"):
        return "Memulai bagian halaman yang dilihat pengguna."
    if code.startswith("<link"):
        return "Menghubungkan file tampilan/style untuk mempercantik halaman."
    if code.startswith("<script"):
        return "Memuat script agar halaman bisa berinteraksi dengan pengguna."
    if code.startswith("<form"):
        return "Membuat area input untuk upload citra dan pengaturan prediksi."
    if code.startswith("<input"):
        return "Membuat kolom input pengguna, seperti file atau threshold."
    if code.startswith("<button"):
        return "Membuat tombol yang bisa diklik pengguna."
    if code.startswith("<img"):
        return "Menampilkan gambar, seperti heatmap, overlay, atau preview."
    if code.startswith("<table"):
        return "Membuat tabel untuk menampilkan riwayat atau ringkasan data."
    if code.startswith("</"):
        return "Menutup bagian halaman yang sudah dibuka sebelumnya."
    return "Menyusun tampilan halaman sistem web."


def explain_css(code: str) -> str:
    if code.startswith("/*"):
        return "Catatan CSS untuk menjelaskan pengaturan tampilan."
    if code.startswith("@media"):
        return "Mengatur tampilan agar tetap rapi di ukuran layar tertentu."
    if code.startswith("@"):
        return "Menggunakan aturan khusus untuk tampilan halaman."
    if code.endswith("{"):
        return "Memulai pengaturan tampilan untuk elemen tertentu."
    if code == "}":
        return "Menutup bagian pengaturan tampilan."
    if ":" in code:
        prop = code.split(":", 1)[0].strip()
        return f"Mengatur tampilan `{prop}` pada elemen halaman."
    return "Mengatur tampilan visual dashboard."


def explain_shell(code: str) -> str:
    if code.startswith("#!"):
        return "Menentukan program yang menjalankan script ini."
    if code.startswith("#"):
        return "Catatan pada script terminal."
    if code.startswith("set "):
        return "Mengatur cara script terminal dijalankan."
    if "python" in code.lower():
        return "Menjalankan script Python untuk training, evaluasi, atau dokumentasi."
    if code.endswith("\\"):
        return "Melanjutkan perintah ke baris berikutnya agar mudah dibaca."
    return "Menjalankan perintah terminal untuk kebutuhan project."


def explain_batch(code: str) -> str:
    lower = code.lower()
    if lower.startswith("rem"):
        return "Catatan pada script Windows."
    if lower.startswith("@echo"):
        return "Mengatur apakah perintah ditampilkan saat script berjalan."
    if lower.startswith("set "):
        return "Mengatur nilai sementara yang dibutuhkan proses Windows."
    if "python" in lower:
        return "Menjalankan script Python lewat file Windows."
    return "Menjalankan perintah Windows untuk kebutuhan project."


def explain_line(code: str, style: LanguageStyle) -> str:
    if style.name == "python":
        return explain_python(code)
    if style.name == "javascript":
        return explain_javascript(code)
    if style.name == "html":
        return explain_html(code)
    if style.name == "css":
        return explain_css(code)
    if style.name == "shell":
        return explain_shell(code)
    if style.name == "batch":
        return explain_batch(code)
    return "Menjalankan baris kode sesuai fungsi file."


def comment_line(text: str, style: LanguageStyle, indent: str) -> str:
    if style.line_prefix:
        separator = " " if style.line_prefix != "REM" else " "
        return f"{indent}{style.line_prefix}{separator}{text}"
    return f"{indent}{style.block_start} {text} {style.block_end}"


def output_path_for(source: Path) -> Path:
    relative = source.relative_to(ROOT)
    suffix = source.suffix
    output_name = source.name[: -len(suffix)] + f".annotated{suffix}" if suffix else source.name + ".annotated"
    return OUTPUT_ROOT / relative.parent / output_name


def thesis_context_for(source: Path) -> str:
    relative = source.relative_to(ROOT).as_posix().lower()
    name = source.name.lower()

    if "data_understanding" in name:
        return "Konteks laporan: Data Understanding, yaitu memahami dataset citra hotspot."
    if "data_preparation" in name or "preprocess" in name:
        return "Konteks laporan: Data Preparation, yaitu menyiapkan citra menjadi mask, sequence, dan patch."
    if "modeling" in name or "train" in name or "convlstm" in name or "run_train" in name:
        return "Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM."
    if "evaluation" in name or "validate" in name or "metric" in name or "geospatial" in name:
        return "Konteks laporan: Evaluation, yaitu mengukur hasil prediksi dan validasi."
    if "backend/" in relative or "frontend/" in relative or "static/" in relative or "deployment" in name:
        return "Konteks laporan: Deployment, yaitu menjalankan model di sistem web."
    if "test" in relative:
        return "Konteks laporan: Pengujian sistem, yaitu memastikan fitur berjalan benar."
    return "Konteks laporan: kode pendukung project skripsi."


def annotate_file(source: Path) -> Path:
    style = STYLE_BY_SUFFIX[source.suffix.lower()]
    text = source.read_text(encoding="utf-8", errors="replace")
    output_lines: list[str] = []
    inside_python_docstring = False

    relative = source.relative_to(ROOT).as_posix()
    output_lines.append(comment_line(f"File anotasi dari `{relative}`.", style, ""))
    output_lines.append(comment_line("File ini untuk belajar/lampiran, bukan source utama aplikasi.", style, ""))
    output_lines.append(comment_line(thesis_context_for(source), style, ""))
    output_lines.append(comment_line("Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.", style, ""))

    for raw_line in text.splitlines():
        code = clean_code(raw_line)
        if not code:
            output_lines.append("")
            continue
        indent = leading_whitespace(raw_line)
        explanation = explain_line(code, style)
        if style.name == "python" and inside_python_docstring and not (
            code.startswith('"""') or code.startswith("'''")
        ):
            explanation = "Isi catatan penjelasan pada bagian kode ini."
        output_lines.append(comment_line(explanation, style, indent))
        output_lines.append(raw_line)
        if style.name == "python" and (code.startswith('"""') or code.startswith("'''")):
            quote = '"""' if code.startswith('"""') else "'''"
            if code.count(quote) == 1:
                inside_python_docstring = not inside_python_docstring

    destination = output_path_for(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    code_files = discover_code_files()
    outputs = [annotate_file(path) for path in code_files]

    index_lines = [
        "# Index Kode Terkomentar Line-by-Line",
        "",
        "Folder ini berisi salinan kode teranotasi untuk membantu memahami isi project.",
        "File di folder ini bukan source utama dan tidak digunakan oleh aplikasi.",
        "Komentar dibuat dengan bahasa sederhana agar mudah dipakai untuk belajar, hafalan, dan lampiran skripsi.",
        "Label konteks laporan membantu menghubungkan kode dengan CRISP-DM: Data Understanding, Data Preparation, Modeling, Evaluation, dan Deployment.",
        "",
        f"Jumlah file teranotasi: {len(outputs)}",
        "",
        "## Daftar File",
        "",
    ]
    for output in outputs:
        index_lines.append(f"- `{output.relative_to(OUTPUT_ROOT).as_posix()}`")
    (OUTPUT_ROOT / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"[annotate] File diproses: {len(outputs)}")
    print(f"[annotate] Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
