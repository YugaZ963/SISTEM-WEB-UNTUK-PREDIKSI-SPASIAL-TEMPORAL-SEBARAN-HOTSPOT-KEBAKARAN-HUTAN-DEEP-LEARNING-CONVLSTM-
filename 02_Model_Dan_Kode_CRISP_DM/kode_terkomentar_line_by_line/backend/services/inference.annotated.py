# File anotasi dari `backend/services/inference.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Deployment model: memuat ConvLSTM, menjalankan inferensi patch_stitch, dan fallback heuristic bila model gagal dimuat.

# Isi catatan penjelasan pada bagian kode ini.
Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
# Isi catatan penjelasan pada bagian kode ini.
terutama data understanding, data preparation, modeling, evaluation,
# Isi catatan penjelasan pada bagian kode ini.
atau deployment sesuai fungsi teknisnya.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import json
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import tempfile
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import zipfile
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from pathlib import Path

# Mengambil alat bantu/library yang diperlukan oleh file ini.
import numpy as np

# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend import settings
# Mengambil bagian tertentu dari library agar bisa dipakai di file ini.
from backend.services.preprocess import ModelSpec


# Membuat wadah bernama `InferenceService` untuk menyimpan data atau aturan kerja.
class InferenceService:
    # Membuat langkah kerja bernama `__init__`.
    def __init__(self, spec: ModelSpec) -> None:
        # Menyimpan nilai ke `self.spec` untuk dipakai pada langkah berikutnya.
        self.spec = spec
        # Menyimpan nilai ke `self.model` untuk dipakai pada langkah berikutnya.
        self.model = None
        # Menyimpan nilai ke `self.model_path` untuk dipakai pada langkah berikutnya.
        self.model_path: str | None = None
        # Menyimpan nilai ke `self.preprocess_mode` untuk dipakai pada langkah berikutnya.
        self.preprocess_mode = spec.preprocess_mode
        # Menyimpan nilai ke `self.backend` untuk dipakai pada langkah berikutnya.
        self.backend = "heuristic-temporal-v1"

    # Membuat langkah kerja bernama `load_model`.
    def load_model(self) -> None:
        # Menyimpan nilai ke `self.model` untuk dipakai pada langkah berikutnya.
        self.model = None
        # Menyimpan nilai ke `self.model_path` untuk dipakai pada langkah berikutnya.
        self.model_path = None
        # Menyimpan nilai ke `self.preprocess_mode` untuk dipakai pada langkah berikutnya.
        self.preprocess_mode = self.spec.preprocess_mode
        # Menyimpan nilai ke `self.backend` untuk dipakai pada langkah berikutnya.
        self.backend = "heuristic-temporal-v1"

        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Mengambil alat bantu/library yang diperlukan oleh file ini.
            import tensorflow as tf  # type: ignore
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception:
            # Menyimpan nilai ke `self.backend` untuk dipakai pada langkah berikutnya.
            self.backend = "heuristic-temporal-v1 (tensorflow-not-installed)"
            # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
            return

        # Menyimpan nilai ke `candidates` untuk dipakai pada langkah berikutnya.
        candidates = self._discover_model_candidates()
        # Mengecek syarat sebelum melanjutkan proses.
        if not candidates:
            # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
            return

        # Mengulang proses untuk setiap data dalam daftar.
        for candidate in candidates:
            # Menyimpan nilai ke `load_result` untuk dipakai pada langkah berikutnya.
            load_result = self._load_candidate_model(candidate=candidate, tf=tf)
            # Mengecek syarat sebelum melanjutkan proses.
            if load_result is None:
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue

            # Menyimpan nilai ke `model, backend_name` untuk dipakai pada langkah berikutnya.
            model, backend_name = load_result
            # Mengecek syarat sebelum melanjutkan proses.
            if not self._model_accepts_spec(model):
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue

            # Menyimpan nilai ke `self.model` untuk dipakai pada langkah berikutnya.
            self.model = model
            # Menyimpan nilai ke `self.model_path` untuk dipakai pada langkah berikutnya.
            self.model_path = str(candidate)
            # Menyimpan nilai ke `self.backend` untuk dipakai pada langkah berikutnya.
            self.backend = backend_name
            # Menyimpan nilai ke `self.preprocess_mode` untuk dipakai pada langkah berikutnya.
            self.preprocess_mode = self.spec.preprocess_mode
            # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
            return

        # Menyimpan nilai ke `self.model` untuk dipakai pada langkah berikutnya.
        self.model = None
        # Menyimpan nilai ke `self.model_path` untuk dipakai pada langkah berikutnya.
        self.model_path = None
        # Menyimpan nilai ke `self.preprocess_mode` untuk dipakai pada langkah berikutnya.
        self.preprocess_mode = self.spec.preprocess_mode
        # Menyimpan nilai ke `self.backend` untuk dipakai pada langkah berikutnya.
        self.backend = "heuristic-temporal-v1 (model-load-failed)"

    # Membuat langkah kerja bernama `_discover_model_candidates`.
    def _discover_model_candidates(self) -> list[Path]:
        # Menyimpan nilai ke `candidates` untuk dipakai pada langkah berikutnya.
        candidates: list[Path] = []
        # Menyimpan nilai ke `seen` untuk dipakai pada langkah berikutnya.
        seen: set[str] = set()

        # Membuat langkah kerja bernama `add_candidate`.
        def add_candidate(path: Path) -> None:
            # Menyimpan nilai ke `key` untuk dipakai pada langkah berikutnya.
            key = str(path.resolve())
            # Mengecek syarat sebelum melanjutkan proses.
            if key in seen:
                # Mengakhiri langkah kerja tanpa mengirim nilai khusus.
                return
            # Melanjutkan langkah kerja pada bagian kode ini.
            seen.add(key)
            # Melanjutkan langkah kerja pada bagian kode ini.
            candidates.append(path)

        # Mengulang proses untuk setiap data dalam daftar.
        for path in settings.CONVLSTM_MODEL_CANDIDATES:
            # Mengecek syarat sebelum melanjutkan proses.
            if path.exists():
                # Melanjutkan langkah kerja pada bagian kode ini.
                add_candidate(path)

        # Mengecek syarat sebelum melanjutkan proses.
        if settings.ACTIVE_MODEL_CONFIG.get("discover_ipynb_models") and settings.IPYNB_DIR.exists():
            # Menyimpan nilai ke `ignored_dirs` untuk dipakai pada langkah berikutnya.
            ignored_dirs = {".venv", "__pycache__", ".ipynb_checkpoints", "backups"}

            # Menyimpan nilai ke `discovered_files` untuk dipakai pada langkah berikutnya.
            discovered_files: list[Path] = []
            # Mengulang proses untuk setiap data dalam daftar.
            for pattern in ("*.keras", "*.h5"):
                # Mengulang proses untuk setiap data dalam daftar.
                for path in settings.IPYNB_DIR.rglob(pattern):
                    # Mengecek syarat sebelum melanjutkan proses.
                    if not path.is_file():
                        # Menyebutkan item yang ikut dipakai pada daftar di atas.
                        continue
                    # Mengecek syarat sebelum melanjutkan proses.
                    if any(part in ignored_dirs for part in path.parts):
                        # Menyebutkan item yang ikut dipakai pada daftar di atas.
                        continue
                    # Melanjutkan langkah kerja pada bagian kode ini.
                    discovered_files.append(path)

            # Menyimpan nilai ke `discovered_files.sort(key` untuk dipakai pada langkah berikutnya.
            discovered_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            # Mengulang proses untuk setiap data dalam daftar.
            for path in discovered_files:
                # Melanjutkan langkah kerja pada bagian kode ini.
                add_candidate(path)

            # Menyimpan nilai ke `discovered_saved_models` untuk dipakai pada langkah berikutnya.
            discovered_saved_models: list[Path] = []
            # Mengulang proses untuk setiap data dalam daftar.
            for pb_file in settings.IPYNB_DIR.rglob("saved_model.pb"):
                # Mengecek syarat sebelum melanjutkan proses.
                if not pb_file.is_file():
                    # Menyebutkan item yang ikut dipakai pada daftar di atas.
                    continue
                # Mengecek syarat sebelum melanjutkan proses.
                if any(part in ignored_dirs for part in pb_file.parts):
                    # Menyebutkan item yang ikut dipakai pada daftar di atas.
                    continue
                # Melanjutkan langkah kerja pada bagian kode ini.
                discovered_saved_models.append(pb_file.parent)

            # Menyimpan nilai ke `discovered_saved_models.sort(key` untuk dipakai pada langkah berikutnya.
            discovered_saved_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            # Mengulang proses untuk setiap data dalam daftar.
            for path in discovered_saved_models:
                # Melanjutkan langkah kerja pada bagian kode ini.
                add_candidate(path)

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return candidates

    # Membuat langkah kerja bernama `list_model_candidates`.
    def list_model_candidates(self) -> list[dict]:
        # Menyimpan nilai ke `records` untuk dipakai pada langkah berikutnya.
        records: list[dict] = []
        # Mengulang proses untuk setiap data dalam daftar.
        for path in settings.CONVLSTM_MODEL_CANDIDATES:
            # Melanjutkan langkah kerja pada bagian kode ini.
            records.append({"path": str(path), "exists": path.exists(), "source": "configured"})

        # Mengulang proses untuk setiap data dalam daftar.
        for path in self._discover_model_candidates():
            # Mengecek syarat sebelum melanjutkan proses.
            if any(item["path"] == str(path) for item in records):
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Melanjutkan langkah kerja pada bagian kode ini.
            records.append({"path": str(path), "exists": True, "source": "discovered"})

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return records

    # Membuat langkah kerja bernama `_load_candidate_model`.
    def _load_candidate_model(self, candidate: Path, tf):
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
            model = tf.keras.models.load_model(candidate, compile=False)
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return model, f"convlstm-tensorflow ({candidate.name})"
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception:
            # Menyimpan nilai ke `compat_model` untuk dipakai pada langkah berikutnya.
            compat_model = self._load_keras_archive_with_compat_weights(candidate=candidate, tf=tf)
            # Mengecek syarat sebelum melanjutkan proses.
            if compat_model is not None:
                # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
                return compat_model, f"convlstm-tensorflow (compat-weights {candidate.name})"
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return None

    # Membuat langkah kerja bernama `_model_accepts_spec`.
    def _model_accepts_spec(self, model) -> bool:
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Menyimpan nilai ke `input_shape` untuk dipakai pada langkah berikutnya.
            input_shape = model.input_shape
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return False

        # Mengecek syarat sebelum melanjutkan proses.
        if isinstance(input_shape, list):
            # Menyimpan nilai ke `input_shape` untuk dipakai pada langkah berikutnya.
            input_shape = input_shape[0]

        # Mengecek syarat sebelum melanjutkan proses.
        if len(input_shape) != 5:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return False

        # Menyimpan nilai ke `expected` untuk dipakai pada langkah berikutnya.
        expected = (
            # Menyebutkan item yang ikut dipakai pada daftar di atas.
            None,
            # Melanjutkan langkah kerja pada bagian kode ini.
            self.spec.time_steps,
            # Melanjutkan langkah kerja pada bagian kode ini.
            self.spec.grid_size,
            # Melanjutkan langkah kerja pada bagian kode ini.
            self.spec.grid_size,
            # Melanjutkan langkah kerja pada bagian kode ini.
            self.spec.channels,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Mengulang proses untuk setiap data dalam daftar.
        for actual, expected_value in zip(input_shape, expected):
            # Mengecek syarat sebelum melanjutkan proses.
            if expected_value is None:
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                continue
            # Mengecek syarat sebelum melanjutkan proses.
            if actual is not None and int(actual) != int(expected_value):
                # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
                return False
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return True

    # Membuat langkah kerja bernama `_load_keras_archive_with_compat_weights`.
    def _load_keras_archive_with_compat_weights(self, candidate: Path, tf):
        # Mengecek syarat sebelum melanjutkan proses.
        if candidate.suffix.lower() != ".keras":
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return None

        # Menyimpan nilai ke `archive_config` untuk dipakai pada langkah berikutnya.
        archive_config = self._read_keras_archive_config(candidate)
        # Mengecek syarat sebelum melanjutkan proses.
        if archive_config is None:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return None

        # Mengecek syarat sebelum melanjutkan proses.
        if not self._is_supported_compat_archive(archive_config):
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return None

        # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
        model = self._build_compat_convlstm_model(tf=tf)
        # Menyimpan nilai ke `weights_path` untuk dipakai pada langkah berikutnya.
        weights_path = self._extract_archive_weights(candidate)
        # Mengecek syarat sebelum melanjutkan proses.
        if weights_path is None:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return None

        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Menyimpan nilai ke `self._assign_compat_weights(model` untuk dipakai pada langkah berikutnya.
            self._assign_compat_weights(model=model, weights_path=weights_path)
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return model
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return None
        # Menjalankan langkah penutup setelah proses percobaan selesai.
        finally:
            # Mencoba menjalankan proses yang mungkin gagal.
            try:
                # Menyimpan nilai ke `weights_path.unlink(missing_ok` untuk dipakai pada langkah berikutnya.
                weights_path.unlink(missing_ok=True)
            # Menangani kesalahan agar program tidak langsung berhenti.
            except Exception:
                # Menyebutkan item yang ikut dipakai pada daftar di atas.
                pass

    # Membuat langkah kerja bernama `_read_keras_archive_config`.
    def _read_keras_archive_config(self, candidate: Path) -> dict | None:
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
            with zipfile.ZipFile(candidate, "r") as archive:
                # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
                return json.loads(archive.read("config.json"))
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return None

    # Membuat langkah kerja bernama `_is_supported_compat_archive`.
    def _is_supported_compat_archive(self, config: dict) -> bool:
        # Mengecek syarat sebelum melanjutkan proses.
        if config.get("class_name") != "Sequential":
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return False

        # Menyimpan nilai ke `layers` untuk dipakai pada langkah berikutnya.
        layers = config.get("config", {}).get("layers", [])
        # Menyimpan nilai ke `class_names` untuk dipakai pada langkah berikutnya.
        class_names = [layer.get("class_name") for layer in layers]
        # Menyimpan nilai ke `expected` untuk dipakai pada langkah berikutnya.
        expected = [
            # Melanjutkan langkah kerja pada bagian kode ini.
            "InputLayer",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "ConvLSTM2D",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "BatchNormalization",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Dropout",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "ConvLSTM2D",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "BatchNormalization",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Dropout",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Conv2D",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Conv2D",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]
        # Mengecek syarat sebelum melanjutkan proses.
        if class_names != expected:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return False

        # Menyimpan nilai ke `input_config` untuk dipakai pada langkah berikutnya.
        input_config = layers[0].get("config", {})
        # Menyimpan nilai ke `batch_shape` untuk dipakai pada langkah berikutnya.
        batch_shape = input_config.get("batch_shape") or input_config.get("batch_input_shape")
        # Menyimpan nilai ke `expected_shape` untuk dipakai pada langkah berikutnya.
        expected_shape = [None, self.spec.time_steps, self.spec.grid_size, self.spec.grid_size, self.spec.channels]
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return batch_shape == expected_shape

    # Membuat langkah kerja bernama `_build_compat_convlstm_model`.
    def _build_compat_convlstm_model(self, tf):
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return tf.keras.Sequential(
            # Memulai susunan data atau perintah yang ditulis beberapa baris.
            [
                # Menyimpan nilai ke `tf.keras.layers.Input(shape` untuk dipakai pada langkah berikutnya.
                tf.keras.layers.Input(shape=(self.spec.time_steps, self.spec.grid_size, self.spec.grid_size, self.spec.channels)),
                # Menyimpan nilai ke `tf.keras.layers.ConvLSTM2D(filters` untuk dipakai pada langkah berikutnya.
                tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True),
                # Melanjutkan langkah kerja pada bagian kode ini.
                tf.keras.layers.BatchNormalization(),
                # Melanjutkan langkah kerja pada bagian kode ini.
                tf.keras.layers.Dropout(0.2),
                # Menyimpan nilai ke `tf.keras.layers.ConvLSTM2D(filters` untuk dipakai pada langkah berikutnya.
                tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=False),
                # Melanjutkan langkah kerja pada bagian kode ini.
                tf.keras.layers.BatchNormalization(),
                # Melanjutkan langkah kerja pada bagian kode ini.
                tf.keras.layers.Dropout(0.2),
                # Menyimpan nilai ke `tf.keras.layers.Conv2D(filters` untuk dipakai pada langkah berikutnya.
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
                # Menyimpan nilai ke `tf.keras.layers.Conv2D(filters` untuk dipakai pada langkah berikutnya.
                tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same"),
            # Menutup susunan data atau perintah yang dimulai sebelumnya.
            ]
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Membuat langkah kerja bernama `_extract_archive_weights`.
    def _extract_archive_weights(self, candidate: Path) -> Path | None:
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
            with zipfile.ZipFile(candidate, "r") as archive:
                # Menyimpan nilai ke `weights_bytes` untuk dipakai pada langkah berikutnya.
                weights_bytes = archive.read("model.weights.h5")
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return None

        # Menyimpan nilai ke `temp_file` untuk dipakai pada langkah berikutnya.
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5")
        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Melanjutkan langkah kerja pada bagian kode ini.
            temp_file.write(weights_bytes)
            # Melanjutkan langkah kerja pada bagian kode ini.
            temp_file.flush()
        # Menjalankan langkah penutup setelah proses percobaan selesai.
        finally:
            # Melanjutkan langkah kerja pada bagian kode ini.
            temp_file.close()
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return Path(temp_file.name)

    # Membuat langkah kerja bernama `_assign_compat_weights`.
    def _assign_compat_weights(self, model, weights_path: Path) -> None:
        # Mengambil alat bantu/library yang diperlukan oleh file ini.
        import h5py  # type: ignore

        # Menyimpan nilai ke `dataset_paths` untuk dipakai pada langkah berikutnya.
        dataset_paths = [
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv_lstm2d/cell/vars/0",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv_lstm2d/cell/vars/1",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv_lstm2d/cell/vars/2",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/batch_normalization/vars/0",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/batch_normalization/vars/1",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/batch_normalization/vars/2",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/batch_normalization/vars/3",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv_lstm2d_1/cell/vars/0",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv_lstm2d_1/cell/vars/1",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv_lstm2d_1/cell/vars/2",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/batch_normalization_1/vars/0",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/batch_normalization_1/vars/1",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/batch_normalization_1/vars/2",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/batch_normalization_1/vars/3",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv2d/vars/0",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv2d/vars/1",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv2d_1/vars/0",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "layers/conv2d_1/vars/1",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ]

        # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
        with h5py.File(weights_path, "r") as handle:
            # Menyimpan nilai ke `arrays` untuk dipakai pada langkah berikutnya.
            arrays = [np.array(handle[path]) for path in dataset_paths]

        # Melanjutkan langkah kerja pada bagian kode ini.
        model.layers[0].set_weights(arrays[0:3])
        # Melanjutkan langkah kerja pada bagian kode ini.
        model.layers[1].set_weights(arrays[3:7])
        # Melanjutkan langkah kerja pada bagian kode ini.
        model.layers[3].set_weights(arrays[7:10])
        # Melanjutkan langkah kerja pada bagian kode ini.
        model.layers[4].set_weights(arrays[10:14])
        # Melanjutkan langkah kerja pada bagian kode ini.
        model.layers[6].set_weights(arrays[14:16])
        # Melanjutkan langkah kerja pada bagian kode ini.
        model.layers[7].set_weights(arrays[16:18])

    # Membuat langkah kerja bernama `_to_grayscale`.
    def _to_grayscale(self, sequence: np.ndarray) -> np.ndarray:
        # Menyimpan nilai ke `frames` untuk dipakai pada langkah berikutnya.
        frames = sequence[0]
        # Mengecek syarat sebelum melanjutkan proses.
        if frames.shape[-1] == 1:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return frames[:, :, :, 0]
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.mean(frames, axis=-1)

    # Membuat langkah kerja bernama `_smooth_map`.
    def _smooth_map(self, arr: np.ndarray) -> np.ndarray:
        # Menyimpan nilai ke `padded` untuk dipakai pada langkah berikutnya.
        padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
        # Menyimpan nilai ke `center` untuk dipakai pada langkah berikutnya.
        center = padded[1:-1, 1:-1]
        # Menyimpan nilai ke `north` untuk dipakai pada langkah berikutnya.
        north = padded[:-2, 1:-1]
        # Menyimpan nilai ke `south` untuk dipakai pada langkah berikutnya.
        south = padded[2:, 1:-1]
        # Menyimpan nilai ke `west` untuk dipakai pada langkah berikutnya.
        west = padded[1:-1, :-2]
        # Menyimpan nilai ke `east` untuk dipakai pada langkah berikutnya.
        east = padded[1:-1, 2:]
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return (0.5 * center) + (0.125 * (north + south + west + east))

    # Membuat langkah kerja bernama `_predict_once_heuristic`.
    def _predict_once_heuristic(self, sequence: np.ndarray) -> np.ndarray:
        # Menyimpan nilai ke `grayscale` untuk dipakai pada langkah berikutnya.
        grayscale = self._to_grayscale(sequence)
        # Menyimpan nilai ke `steps` untuk dipakai pada langkah berikutnya.
        steps = grayscale.shape[0]

        # Menyimpan nilai ke `weights` untuk dipakai pada langkah berikutnya.
        weights = np.linspace(1.0, float(steps), num=steps, dtype=np.float32)
        # Menyimpan nilai ke `weights /` untuk dipakai pada langkah berikutnya.
        weights /= np.sum(weights)
        # Menyimpan nilai ke `weighted_history` untuk dipakai pada langkah berikutnya.
        weighted_history = np.tensordot(weights, grayscale, axes=(0, 0))

        # Menyimpan nilai ke `last_frame` untuk dipakai pada langkah berikutnya.
        last_frame = grayscale[-1]
        # Menyimpan nilai ke `previous_frame` untuk dipakai pada langkah berikutnya.
        previous_frame = grayscale[-2]
        # Menyimpan nilai ke `short_trend` untuk dipakai pada langkah berikutnya.
        short_trend = np.clip(last_frame - previous_frame, 0.0, 1.0)

        # Menyimpan nilai ke `raw_probability` untuk dipakai pada langkah berikutnya.
        raw_probability = (
            # Melanjutkan langkah kerja pada bagian kode ini.
            (0.55 * last_frame)
            # Melanjutkan langkah kerja pada bagian kode ini.
            + (0.30 * weighted_history)
            # Melanjutkan langkah kerja pada bagian kode ini.
            + (0.15 * short_trend)
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
        # Menyimpan nilai ke `smoothed` untuk dipakai pada langkah berikutnya.
        smoothed = self._smooth_map(np.clip(raw_probability, 0.0, 1.0))
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.clip(smoothed, 0.0, 1.0).astype(np.float32)

    # Membuat langkah kerja bernama `_predict_once_model`.
    def _predict_once_model(self, sequence: np.ndarray) -> np.ndarray:
        # Mengecek syarat sebelum melanjutkan proses.
        if self.model is None:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise ValueError("Model ConvLSTM belum termuat.")

        # Menyimpan nilai ke `raw` untuk dipakai pada langkah berikutnya.
        raw = self.model.predict(sequence, verbose=0)
        # Menyimpan nilai ke `prediction` untuk dipakai pada langkah berikutnya.
        prediction = np.asarray(raw, dtype=np.float32)

        # Mengecek syarat sebelum melanjutkan proses.
        if prediction.ndim == 5:
            # Menyimpan nilai ke `prediction` untuk dipakai pada langkah berikutnya.
            prediction = prediction[0]
        # Mengecek syarat sebelum melanjutkan proses.
        if prediction.ndim == 4 and prediction.shape[0] == 1:
            # Menyimpan nilai ke `prediction` untuk dipakai pada langkah berikutnya.
            prediction = prediction[0]
        # Mengecek syarat sebelum melanjutkan proses.
        if prediction.ndim == 3 and prediction.shape[-1] == 1:
            # Menyimpan nilai ke `prediction` untuk dipakai pada langkah berikutnya.
            prediction = prediction[:, :, 0]
        # Mengecek syarat sebelum melanjutkan proses.
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            # Menyimpan nilai ke `prediction` untuk dipakai pada langkah berikutnya.
            prediction = prediction[0]

        # Mengecek syarat sebelum melanjutkan proses.
        if prediction.ndim != 2:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise ValueError(f"Output model tidak didukung: {prediction.shape}")

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.clip(prediction, 0.0, 1.0).astype(np.float32)

    # Membuat langkah kerja bernama `_predict_batch_model`.
    def _predict_batch_model(self, batch: np.ndarray) -> np.ndarray:
        # Mengecek syarat sebelum melanjutkan proses.
        if self.model is None:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise ValueError("Model ConvLSTM belum termuat.")

        # Menyimpan nilai ke `raw` untuk dipakai pada langkah berikutnya.
        raw = self.model.predict(batch, verbose=0)
        # Menyimpan nilai ke `prediction` untuk dipakai pada langkah berikutnya.
        prediction = np.asarray(raw, dtype=np.float32)

        # Mengecek syarat sebelum melanjutkan proses.
        if prediction.ndim == 5 and prediction.shape[-1] == 1:
            # Menyimpan nilai ke `prediction` untuk dipakai pada langkah berikutnya.
            prediction = prediction[:, 0, :, :, 0]
        # Mengecek syarat lain jika syarat sebelumnya tidak terpenuhi.
        elif prediction.ndim == 4 and prediction.shape[-1] == 1:
            # Menyimpan nilai ke `prediction` untuk dipakai pada langkah berikutnya.
            prediction = prediction[:, :, :, 0]
        # Mengecek syarat lain jika syarat sebelumnya tidak terpenuhi.
        elif prediction.ndim != 3:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise ValueError(f"Output batch model tidak didukung: {prediction.shape}")

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.clip(prediction, 0.0, 1.0).astype(np.float32)

    # Membuat langkah kerja bernama `_patch_positions`.
    def _patch_positions(self, length: int, patch_size: int, stride: int) -> list[int]:
        # Mengecek syarat sebelum melanjutkan proses.
        if length <= patch_size:
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return [0]

        # Menyimpan nilai ke `positions` untuk dipakai pada langkah berikutnya.
        positions = list(range(0, max(1, length - patch_size + 1), stride))
        # Menyimpan nilai ke `last` untuk dipakai pada langkah berikutnya.
        last = length - patch_size
        # Mengecek syarat sebelum melanjutkan proses.
        if positions[-1] != last:
            # Melanjutkan langkah kerja pada bagian kode ini.
            positions.append(last)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return positions

    # Membuat langkah kerja bernama `_build_patch_batch`.
    def _build_patch_batch(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        self,
        # Menjelaskan data `sequence` yang disimpan atau dikirim pada bagian ini.
        sequence: np.ndarray,
        # Menjelaskan data `patch_size` yang disimpan atau dikirim pada bagian ini.
        patch_size: int,
        # Menjelaskan data `stride` yang disimpan atau dikirim pada bagian ini.
        stride: int,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        # Menyimpan nilai ke `_, height, width` untuk dipakai pada langkah berikutnya.
        _, height, width = sequence.shape
        # Menyimpan nilai ke `y_positions` untuk dipakai pada langkah berikutnya.
        y_positions = self._patch_positions(height, patch_size, stride)
        # Menyimpan nilai ke `x_positions` untuk dipakai pada langkah berikutnya.
        x_positions = self._patch_positions(width, patch_size, stride)

        # Menyimpan nilai ke `padded_height` untuk dipakai pada langkah berikutnya.
        padded_height = max(height, y_positions[-1] + patch_size)
        # Menyimpan nilai ke `padded_width` untuk dipakai pada langkah berikutnya.
        padded_width = max(width, x_positions[-1] + patch_size)
        # Menyimpan nilai ke `pad_height` untuk dipakai pada langkah berikutnya.
        pad_height = padded_height - height
        # Menyimpan nilai ke `pad_width` untuk dipakai pada langkah berikutnya.
        pad_width = padded_width - width
        # Menyimpan nilai ke `padded` untuk dipakai pada langkah berikutnya.
        padded = np.pad(sequence, ((0, 0), (0, pad_height), (0, pad_width)), mode="constant")

        # Menyimpan nilai ke `patches` untuk dipakai pada langkah berikutnya.
        patches: list[np.ndarray] = []
        # Menyimpan nilai ke `coords` untuk dipakai pada langkah berikutnya.
        coords: list[tuple[int, int]] = []
        # Mengulang proses untuk setiap data dalam daftar.
        for y in y_positions:
            # Mengulang proses untuk setiap data dalam daftar.
            for x in x_positions:
                # Menyimpan nilai ke `patch` untuk dipakai pada langkah berikutnya.
                patch = padded[:, y : y + patch_size, x : x + patch_size]
                # Mengecek syarat sebelum melanjutkan proses.
                if self.spec.channels == 1:
                    # Menyimpan nilai ke `patch` untuk dipakai pada langkah berikutnya.
                    patch = np.expand_dims(patch, axis=-1)
                # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
                else:
                    # Menyimpan nilai ke `patch` untuk dipakai pada langkah berikutnya.
                    patch = np.repeat(np.expand_dims(patch, axis=-1), repeats=self.spec.channels, axis=-1)
                # Melanjutkan langkah kerja pada bagian kode ini.
                patches.append(patch)
                # Melanjutkan langkah kerja pada bagian kode ini.
                coords.append((y, x))

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.stack(patches, axis=0).astype(np.float32), coords

    # Membuat langkah kerja bernama `_predict_patch_batch`.
    def _predict_patch_batch(self, batch: np.ndarray) -> np.ndarray:
        # Mengecek syarat sebelum melanjutkan proses.
        if self.model is None:
            # Menyimpan nilai ke `last_frame` untuk dipakai pada langkah berikutnya.
            last_frame = batch[:, -1, :, :, 0]
            # Menyimpan nilai ke `weighted` untuk dipakai pada langkah berikutnya.
            weighted = np.mean(batch[:, :, :, :, 0], axis=1)
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return np.clip((0.7 * last_frame) + (0.3 * weighted), 0.0, 1.0).astype(np.float32)

        # Menyimpan nilai ke `batch_size` untuk dipakai pada langkah berikutnya.
        batch_size = self.spec.patch_batch_size or 32
        # Menyimpan nilai ke `predictions` untuk dipakai pada langkah berikutnya.
        predictions: list[np.ndarray] = []
        # Mengulang proses untuk setiap data dalam daftar.
        for start in range(0, batch.shape[0], batch_size):
            # Menyimpan nilai ke `chunk` untuk dipakai pada langkah berikutnya.
            chunk = batch[start : start + batch_size]
            # Melanjutkan langkah kerja pada bagian kode ini.
            predictions.append(self._predict_batch_model(chunk))
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.concatenate(predictions, axis=0)

    # Membuat langkah kerja bernama `_stitch_patch_predictions`.
    def _stitch_patch_predictions(
        # Menyebutkan item yang ikut dipakai pada daftar di atas.
        self,
        # Menjelaskan data `predictions` yang disimpan atau dikirim pada bagian ini.
        predictions: np.ndarray,
        # Menjelaskan data `coords` yang disimpan atau dikirim pada bagian ini.
        coords: list[tuple[int, int]],
        # Menjelaskan data `output_shape` yang disimpan atau dikirim pada bagian ini.
        output_shape: tuple[int, int],
        # Menjelaskan data `patch_size` yang disimpan atau dikirim pada bagian ini.
        patch_size: int,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ) -> np.ndarray:
        # Menyimpan nilai ke `height, width` untuk dipakai pada langkah berikutnya.
        height, width = output_shape
        # Menyimpan nilai ke `padded_height` untuk dipakai pada langkah berikutnya.
        padded_height = max(height, max(y for y, _ in coords) + patch_size)
        # Menyimpan nilai ke `padded_width` untuk dipakai pada langkah berikutnya.
        padded_width = max(width, max(x for _, x in coords) + patch_size)
        # Menyimpan nilai ke `canvas` untuk dipakai pada langkah berikutnya.
        canvas = np.zeros((padded_height, padded_width), dtype=np.float32)
        # Menyimpan nilai ke `weights` untuk dipakai pada langkah berikutnya.
        weights = np.zeros((padded_height, padded_width), dtype=np.float32)

        # Mengulang proses untuk setiap data dalam daftar.
        for prediction, (y, x) in zip(predictions, coords):
            # Menyimpan nilai ke `canvas[y` untuk dipakai pada langkah berikutnya.
            canvas[y : y + patch_size, x : x + patch_size] += prediction
            # Menyimpan nilai ke `weights[y` untuk dipakai pada langkah berikutnya.
            weights[y : y + patch_size, x : x + patch_size] += 1.0

        # Menyimpan nilai ke `stitched` untuk dipakai pada langkah berikutnya.
        stitched = canvas / np.maximum(weights, 1e-6)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return np.clip(stitched[:height, :width], 0.0, 1.0).astype(np.float32)

    # Membuat langkah kerja bernama `_predict_once_patch_stitch`.
    def _predict_once_patch_stitch(self, sequence: np.ndarray) -> np.ndarray:
        # Mengecek syarat sebelum melanjutkan proses.
        if sequence.ndim != 3:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise ValueError(f"Sequence patch harus 3D (time, height, width), dapat {sequence.shape}")

        # Menyimpan nilai ke `patch_size` untuk dipakai pada langkah berikutnya.
        patch_size = self.spec.patch_size or self.spec.grid_size
        # Menyimpan nilai ke `stride` untuk dipakai pada langkah berikutnya.
        stride = self.spec.patch_stride or patch_size
        # Menyimpan nilai ke `_, height, width` untuk dipakai pada langkah berikutnya.
        _, height, width = sequence.shape
        # Menyimpan nilai ke `batch, coords` untuk dipakai pada langkah berikutnya.
        batch, coords = self._build_patch_batch(sequence=sequence, patch_size=patch_size, stride=stride)

        # Mencoba menjalankan proses yang mungkin gagal.
        try:
            # Menyimpan nilai ke `predictions` untuk dipakai pada langkah berikutnya.
            predictions = self._predict_patch_batch(batch)
        # Menangani kesalahan agar program tidak langsung berhenti.
        except Exception:
            # Menyimpan nilai ke `self.model` untuk dipakai pada langkah berikutnya.
            self.model = None
            # Menyimpan nilai ke `self.model_path` untuk dipakai pada langkah berikutnya.
            self.model_path = None
            # Menyimpan nilai ke `self.backend` untuk dipakai pada langkah berikutnya.
            self.backend = "heuristic-temporal-v1 (runtime-fallback)"
            # Menyimpan nilai ke `predictions` untuk dipakai pada langkah berikutnya.
            predictions = self._predict_patch_batch(batch)

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return self._stitch_patch_predictions(
            # Menyimpan nilai ke `predictions` untuk dipakai pada langkah berikutnya.
            predictions=predictions,
            # Menyimpan nilai ke `coords` untuk dipakai pada langkah berikutnya.
            coords=coords,
            # Menyimpan nilai ke `output_shape` untuk dipakai pada langkah berikutnya.
            output_shape=(height, width),
            # Menyimpan nilai ke `patch_size` untuk dipakai pada langkah berikutnya.
            patch_size=patch_size,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )

    # Membuat langkah kerja bernama `_predict_horizon_patch_stitch`.
    def _predict_horizon_patch_stitch(self, sequence: np.ndarray, horizon: int) -> list[np.ndarray]:
        # Menyimpan nilai ke `running` untuk dipakai pada langkah berikutnya.
        running = np.copy(sequence)
        # Menyimpan nilai ke `outputs` untuk dipakai pada langkah berikutnya.
        outputs: list[np.ndarray] = []

        # Mengulang proses untuk setiap data dalam daftar.
        for _ in range(horizon):
            # Menyimpan nilai ke `predicted` untuk dipakai pada langkah berikutnya.
            predicted = self._predict_once_patch_stitch(running)
            # Melanjutkan langkah kerja pada bagian kode ini.
            outputs.append(predicted)
            # Menyimpan nilai ke `running` untuk dipakai pada langkah berikutnya.
            running = np.concatenate([running[1:], np.expand_dims(predicted, axis=0)], axis=0)

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return outputs

    # Membuat langkah kerja bernama `predict_horizon`.
    def predict_horizon(self, sequence: np.ndarray, horizon: int = 1) -> list[np.ndarray]:
        # Mengecek syarat sebelum melanjutkan proses.
        if horizon < 1:
            # Menghentikan proses dan memberi pesan kesalahan yang jelas.
            raise ValueError("Horizon minimal adalah 1.")

        # Mengecek syarat sebelum melanjutkan proses.
        if self.spec.inference_mode == "patch_stitch":
            # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
            return self._predict_horizon_patch_stitch(sequence=sequence, horizon=horizon)

        # Menyimpan nilai ke `running` untuk dipakai pada langkah berikutnya.
        running = np.copy(sequence)
        # Menyimpan nilai ke `outputs` untuk dipakai pada langkah berikutnya.
        outputs: list[np.ndarray] = []
        # Mengulang proses untuk setiap data dalam daftar.
        for _ in range(horizon):
            # Mencoba menjalankan proses yang mungkin gagal.
            try:
                # Mengecek syarat sebelum melanjutkan proses.
                if self.model is not None:
                    # Menyimpan nilai ke `predicted` untuk dipakai pada langkah berikutnya.
                    predicted = self._predict_once_model(running)
                # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
                else:
                    # Menyimpan nilai ke `predicted` untuk dipakai pada langkah berikutnya.
                    predicted = self._predict_once_heuristic(running)
            # Menangani kesalahan agar program tidak langsung berhenti.
            except Exception:
                # Catatan asli dari pembuat kode.
                # Jika runtime model gagal saat inferensi, sistem otomatis fallback.
                # Menyimpan nilai ke `self.model` untuk dipakai pada langkah berikutnya.
                self.model = None
                # Menyimpan nilai ke `self.model_path` untuk dipakai pada langkah berikutnya.
                self.model_path = None
                # Menyimpan nilai ke `self.backend` untuk dipakai pada langkah berikutnya.
                self.backend = "heuristic-temporal-v1 (runtime-fallback)"
                # Menyimpan nilai ke `predicted` untuk dipakai pada langkah berikutnya.
                predicted = self._predict_once_heuristic(running)
            # Melanjutkan langkah kerja pada bagian kode ini.
            outputs.append(predicted)

            # Menyimpan nilai ke `next_frame` untuk dipakai pada langkah berikutnya.
            next_frame = np.expand_dims(predicted, axis=-1)
            # Mengecek syarat sebelum melanjutkan proses.
            if self.spec.channels == 3:
                # Menyimpan nilai ke `next_frame` untuk dipakai pada langkah berikutnya.
                next_frame = np.repeat(next_frame, repeats=3, axis=-1)
            # Menyimpan nilai ke `next_frame` untuk dipakai pada langkah berikutnya.
            next_frame = np.expand_dims(next_frame, axis=0)

            # Menyimpan nilai ke `without_batch` untuk dipakai pada langkah berikutnya.
            without_batch = running[0]
            # Menyimpan nilai ke `shifted` untuk dipakai pada langkah berikutnya.
            shifted = np.concatenate([without_batch[1:], next_frame], axis=0)
            # Menyimpan nilai ke `running` untuk dipakai pada langkah berikutnya.
            running = np.expand_dims(shifted, axis=0)

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return outputs
