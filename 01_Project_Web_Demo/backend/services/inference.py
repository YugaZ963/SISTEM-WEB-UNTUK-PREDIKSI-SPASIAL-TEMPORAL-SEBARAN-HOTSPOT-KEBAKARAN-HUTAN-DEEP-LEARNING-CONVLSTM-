"""Komentar file skripsi:
Deployment model untuk memuat model ConvLSTM dan menjalankan prediksi area risiko hotspot.

Konteks laporan: file ini mendukung BAB IV pada bagian modeling, evaluation, dan deployment,
karena berfungsi sebagai service inferensi yang menghubungkan model ConvLSTM terlatih
dengan sistem web prediksi hotspot H+1.
"""

# json digunakan untuk membaca konfigurasi model dari arsip .keras saat diperlukan mode kompatibilitas.
import json

# tempfile digunakan untuk membuat file sementara saat mengekstrak bobot model dari arsip .keras.
import tempfile

# zipfile digunakan untuk membaca isi file .keras karena format .keras merupakan arsip terkompresi.
import zipfile

# Path digunakan untuk mengelola lokasi file model dan file sementara secara aman.
from pathlib import Path

# NumPy digunakan untuk pemrosesan array input, output, patch, dan probability map.
import numpy as np

# settings berisi konfigurasi model aktif, kandidat model, folder model, dan parameter sistem.
from backend import settings

# ModelSpec berisi spesifikasi model seperti grid size, channel, time step, patch size, dan mode inferensi.
from backend.services.preprocess import ModelSpec


# Class utama service inferensi yang bertugas memuat model dan menjalankan prediksi.
class InferenceService:
    # Konstruktor menerima spesifikasi model aktif dari main.py/settings.py.
    def __init__(self, spec: ModelSpec) -> None:
        # Menyimpan spesifikasi model agar dapat dipakai di seluruh proses inferensi.
        self.spec = spec

        # Menyiapkan atribut model; default None sampai model berhasil dimuat.
        self.model = None

        # Menyimpan path model yang berhasil dimuat; default None jika model belum/gagal dimuat.
        self.model_path: str | None = None

        # Menyimpan mode preprocessing yang dipakai, mengikuti konfigurasi ModelSpec.
        self.preprocess_mode = spec.preprocess_mode

        # Backend default adalah heuristic agar sistem tetap berjalan jika model/TensorFlow belum tersedia.
        self.backend = "heuristic-temporal-v1"

    # Fungsi untuk memuat model ConvLSTM dari kandidat model yang tersedia.
    def load_model(self) -> None:
        # Reset object model sebelum proses load dilakukan.
        self.model = None

        # Reset path model agar tidak memakai path lama.
        self.model_path = None

        # Reset mode preprocessing sesuai spesifikasi aktif.
        self.preprocess_mode = self.spec.preprocess_mode

        # Reset backend ke heuristic sebelum model berhasil dimuat.
        self.backend = "heuristic-temporal-v1"

        # TensorFlow diimport di dalam fungsi agar aplikasi tetap bisa berjalan walaupun TensorFlow belum terinstall.
        try:
            # Import TensorFlow sebagai library untuk memuat model Keras/ConvLSTM.
            import tensorflow as tf  # type: ignore

        # Jika TensorFlow tidak tersedia, sistem memakai fallback heuristic.
        except Exception:
            # Menandai bahwa fallback terjadi karena TensorFlow belum terinstall.
            self.backend = "heuristic-temporal-v1 (tensorflow-not-installed)"

            # Menghentikan proses load model karena TensorFlow tidak tersedia.
            return

        # Mengambil daftar kandidat model yang tersedia di folder konfigurasi.
        candidates = self._discover_model_candidates()

        # Jika tidak ada kandidat model, service tetap memakai heuristic.
        if not candidates:
            # Keluar tanpa error agar dashboard tetap dapat berjalan.
            return

        # Mencoba setiap kandidat model satu per satu.
        for candidate in candidates:
            # Mencoba memuat model dari kandidat.
            load_result = self._load_candidate_model(candidate=candidate, tf=tf)

            # Jika kandidat gagal dimuat, lanjut ke kandidat berikutnya.
            if load_result is None:
                continue

            # Membongkar hasil load menjadi object model dan nama backend.
            model, backend_name = load_result

            # Memastikan shape input model sesuai spesifikasi sistem aktif.
            if not self._model_accepts_spec(model):
                # Jika shape tidak cocok, kandidat dilewati.
                continue

            # Menyimpan model yang berhasil dimuat.
            self.model = model

            # Menyimpan path model yang berhasil dipakai.
            self.model_path = str(candidate)

            # Menyimpan nama backend ConvLSTM TensorFlow.
            self.backend = backend_name

            # Menyimpan mode preprocessing sesuai spesifikasi model.
            self.preprocess_mode = self.spec.preprocess_mode

            # Menghentikan proses karena model valid sudah ditemukan.
            return

        # Jika seluruh kandidat gagal, model dikosongkan.
        self.model = None

        # Path model juga dikosongkan.
        self.model_path = None

        # Preprocessing tetap mengikuti spesifikasi aktif.
        self.preprocess_mode = self.spec.preprocess_mode

        # Backend diberi keterangan bahwa model gagal dimuat.
        self.backend = "heuristic-temporal-v1 (model-load-failed)"

    # Fungsi untuk menemukan kandidat model berdasarkan konfigurasi settings.
    def _discover_model_candidates(self) -> list[Path]:
        # Menyiapkan list kandidat model.
        candidates: list[Path] = []

        # Set dipakai agar kandidat model tidak duplikat.
        seen: set[str] = set()

        # Fungsi lokal untuk menambahkan kandidat model secara unik.
        def add_candidate(path: Path) -> None:
            # Mengubah path menjadi path absolut sebagai key unik.
            key = str(path.resolve())

            # Jika path sudah pernah ditambahkan, abaikan.
            if key in seen:
                return

            # Menandai path sebagai sudah dilihat.
            seen.add(key)

            # Menambahkan path ke daftar kandidat.
            candidates.append(path)

        # Membaca kandidat model yang dikonfigurasi langsung pada settings.
        for path in settings.CONVLSTM_MODEL_CANDIDATES:
            # Hanya model yang benar-benar ada di file system yang ditambahkan.
            if path.exists():
                add_candidate(path)

        # Jika profile aktif mengizinkan pencarian model dari folder Ipynb, lakukan discovery tambahan.
        if settings.ACTIVE_MODEL_CONFIG.get("discover_ipynb_models") and settings.IPYNB_DIR.exists():
            # Folder yang diabaikan agar pencarian tidak membaca virtual environment/cache.
            ignored_dirs = {".venv", "__pycache__", ".ipynb_checkpoints", "backups"}

            # Menyiapkan list file model yang ditemukan.
            discovered_files: list[Path] = []

            # Mencari file model berformat .keras dan .h5.
            for pattern in ("*.keras", "*.h5"):
                # Melakukan pencarian rekursif di folder Ipynb.
                for path in settings.IPYNB_DIR.rglob(pattern):
                    # Melewati path yang bukan file.
                    if not path.is_file():
                        continue

                    # Melewati file yang berada di folder yang diabaikan.
                    if any(part in ignored_dirs for part in path.parts):
                        continue

                    # Menambahkan file model yang ditemukan.
                    discovered_files.append(path)

            # Mengurutkan model berdasarkan waktu modifikasi terbaru.
            discovered_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Menambahkan file model yang ditemukan ke daftar kandidat.
            for path in discovered_files:
                add_candidate(path)

            # Menyiapkan list folder SavedModel yang ditemukan.
            discovered_saved_models: list[Path] = []

            # Mencari file saved_model.pb sebagai penanda folder SavedModel.
            for pb_file in settings.IPYNB_DIR.rglob("saved_model.pb"):
                # Melewati path yang bukan file.
                if not pb_file.is_file():
                    continue

                # Melewati folder yang diabaikan.
                if any(part in ignored_dirs for part in pb_file.parts):
                    continue

                # Menambahkan folder induk saved_model.pb sebagai kandidat SavedModel.
                discovered_saved_models.append(pb_file.parent)

            # Mengurutkan folder SavedModel berdasarkan waktu modifikasi terbaru.
            discovered_saved_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Menambahkan folder SavedModel ke kandidat.
            for path in discovered_saved_models:
                add_candidate(path)

        # Mengembalikan seluruh kandidat model.
        return candidates

    # Fungsi untuk menampilkan daftar kandidat model pada endpoint runtime/status.
    def list_model_candidates(self) -> list[dict]:
        # Menyiapkan list record kandidat model.
        records: list[dict] = []

        # Mendaftarkan kandidat model yang dikonfigurasi langsung.
        for path in settings.CONVLSTM_MODEL_CANDIDATES:
            # Menyimpan path, status keberadaan, dan sumber kandidat.
            records.append({"path": str(path), "exists": path.exists(), "source": "configured"})

        # Mendaftarkan kandidat model hasil discovery.
        for path in self._discover_model_candidates():
            # Jika sudah ada pada record configured, tidak perlu ditambahkan ulang.
            if any(item["path"] == str(path) for item in records):
                continue

            # Menambahkan kandidat discovery ke daftar record.
            records.append({"path": str(path), "exists": True, "source": "discovered"})

        # Mengembalikan daftar kandidat model.
        return records

    # Fungsi untuk memuat satu kandidat model.
    def _load_candidate_model(self, candidate: Path, tf):
        # Mencoba load model langsung menggunakan TensorFlow Keras.
        try:
            # compile=False dipakai karena inferensi tidak memerlukan kompilasi training.
            model = tf.keras.models.load_model(candidate, compile=False)

            # Mengembalikan model dan nama backend jika load berhasil.
            return model, f"convlstm-tensorflow ({candidate.name})"

        # Jika load biasa gagal, sistem mencoba mode kompatibilitas bobot.
        except Exception:
            # Mencoba membaca arsip .keras dan membangun ulang model secara kompatibel.
            compat_model = self._load_keras_archive_with_compat_weights(candidate=candidate, tf=tf)

            # Jika mode kompatibel berhasil, model dikembalikan.
            if compat_model is not None:
                return compat_model, f"convlstm-tensorflow (compat-weights {candidate.name})"

        # Jika semua cara gagal, kandidat dianggap tidak valid.
        return None

    # Fungsi untuk memastikan shape input model sesuai ModelSpec.
    def _model_accepts_spec(self, model) -> bool:
        # Mencoba membaca input_shape model.
        try:
            # input_shape berisi format input yang diterima model.
            input_shape = model.input_shape

        # Jika input_shape tidak dapat dibaca, model tidak diterima.
        except Exception:
            return False

        # Jika model memiliki lebih dari satu input, ambil input pertama.
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        # Model ConvLSTM harus memiliki 5 dimensi: batch, time, height, width, channel.
        if len(input_shape) != 5:
            return False

        # Shape yang diharapkan berdasarkan spesifikasi model aktif.
        expected = (
            None,
            self.spec.time_steps,
            self.spec.grid_size,
            self.spec.grid_size,
            self.spec.channels,
        )

        # Membandingkan setiap dimensi shape aktual dan shape yang diharapkan.
        for actual, expected_value in zip(input_shape, expected):
            # None pada expected berarti dimensi batch fleksibel.
            if expected_value is None:
                continue

            # Jika dimensi aktual tidak cocok, model tidak valid.
            if actual is not None and int(actual) != int(expected_value):
                return False

        # Model diterima jika seluruh dimensi sesuai.
        return True

    # Fungsi untuk memuat model .keras dengan cara kompatibilitas jika load_model biasa gagal.
    def _load_keras_archive_with_compat_weights(self, candidate: Path, tf):
        # Mode kompatibilitas hanya berlaku untuk file .keras.
        if candidate.suffix.lower() != ".keras":
            return None

        # Membaca konfigurasi model dari arsip .keras.
        archive_config = self._read_keras_archive_config(candidate)

        # Jika config tidak bisa dibaca, proses gagal.
        if archive_config is None:
            return None

        # Memastikan struktur arsip cocok dengan arsitektur ConvLSTM yang didukung.
        if not self._is_supported_compat_archive(archive_config):
            return None

        # Membangun ulang arsitektur ConvLSTM yang kompatibel.
        model = self._build_compat_convlstm_model(tf=tf)

        # Mengekstrak bobot dari arsip .keras ke file sementara.
        weights_path = self._extract_archive_weights(candidate)

        # Jika bobot tidak bisa diekstrak, proses gagal.
        if weights_path is None:
            return None

        # Mencoba memasangkan bobot ke model kompatibel.
        try:
            # Assign bobot ke layer-layer model.
            self._assign_compat_weights(model=model, weights_path=weights_path)

            # Mengembalikan model jika bobot berhasil dipasang.
            return model

        # Jika pemasangan bobot gagal, proses kompatibilitas gagal.
        except Exception:
            return None

        # File bobot sementara dibersihkan setelah proses selesai.
        finally:
            try:
                # Menghapus file temporary weights jika masih ada.
                weights_path.unlink(missing_ok=True)

            # Error penghapusan file sementara diabaikan agar tidak mengganggu aplikasi.
            except Exception:
                pass

    # Fungsi untuk membaca config.json dari arsip .keras.
    def _read_keras_archive_config(self, candidate: Path) -> dict | None:
        # Mencoba membuka file .keras sebagai zip archive.
        try:
            # Membuka arsip .keras dalam mode read.
            with zipfile.ZipFile(candidate, "r") as archive:
                # Membaca config.json dan mengubahnya menjadi dictionary.
                return json.loads(archive.read("config.json"))

        # Jika arsip/config tidak bisa dibaca, kembalikan None.
        except Exception:
            return None

    # Fungsi untuk memeriksa apakah arsip .keras cocok dengan arsitektur kompatibel.
    def _is_supported_compat_archive(self, config: dict) -> bool:
        # Mode kompatibel hanya mendukung model Sequential.
        if config.get("class_name") != "Sequential":
            return False

        # Mengambil daftar layer dari konfigurasi model.
        layers = config.get("config", {}).get("layers", [])

        # Mengambil nama class setiap layer.
        class_names = [layer.get("class_name") for layer in layers]

        # Struktur layer yang diharapkan untuk model ConvLSTM pada penelitian ini.
        expected = [
            "InputLayer",
            "ConvLSTM2D",
            "BatchNormalization",
            "Dropout",
            "ConvLSTM2D",
            "BatchNormalization",
            "Dropout",
            "Conv2D",
            "Conv2D",
        ]

        # Jika urutan layer berbeda, arsip tidak didukung.
        if class_names != expected:
            return False

        # Mengambil konfigurasi input layer.
        input_config = layers[0].get("config", {})

        # Membaca batch shape dari config model.
        batch_shape = input_config.get("batch_shape") or input_config.get("batch_input_shape")

        # Menentukan shape input yang sesuai dengan ModelSpec.
        expected_shape = [None, self.spec.time_steps, self.spec.grid_size, self.spec.grid_size, self.spec.channels]

        # Mengembalikan True jika batch shape sesuai.
        return batch_shape == expected_shape

    # Fungsi untuk membangun ulang arsitektur ConvLSTM yang kompatibel dengan bobot arsip.
    def _build_compat_convlstm_model(self, tf):
        # Mengembalikan model Sequential dengan arsitektur ConvLSTM dua lapis dan output sigmoid.
        return tf.keras.Sequential(
            [
                # Input menerima sequence citra dengan bentuk time, height, width, channel.
                tf.keras.layers.Input(shape=(self.spec.time_steps, self.spec.grid_size, self.spec.grid_size, self.spec.channels)),
                # ConvLSTM pertama mempelajari pola spasial-temporal dan mengembalikan sequence.
                tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True),
                # BatchNormalization menstabilkan distribusi aktivasi.
                tf.keras.layers.BatchNormalization(),
                # Dropout mengurangi risiko overfitting.
                tf.keras.layers.Dropout(0.2),
                # ConvLSTM kedua merangkum pola temporal menjadi satu representasi spasial.
                tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=False),
                # BatchNormalization kedua untuk stabilisasi output ConvLSTM.
                tf.keras.layers.BatchNormalization(),
                # Dropout kedua untuk regularisasi.
                tf.keras.layers.Dropout(0.2),
                # Conv2D 3x3 membentuk fitur spasial akhir.
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
                # Conv2D 1x1 sigmoid menghasilkan probability map risiko hotspot 0 sampai 1.
                tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same"),
            ]
        )

    # Fungsi untuk mengekstrak file bobot dari arsip .keras.
    def _extract_archive_weights(self, candidate: Path) -> Path | None:
        # Mencoba membaca file model.weights.h5 dari arsip.
        try:
            # Membuka arsip .keras.
            with zipfile.ZipFile(candidate, "r") as archive:
                # Membaca isi file bobot.
                weights_bytes = archive.read("model.weights.h5")

        # Jika bobot tidak ada atau gagal dibaca, proses gagal.
        except Exception:
            return None

        # Membuat file sementara untuk menyimpan bobot.
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5")

        # Menulis bobot ke file sementara.
        try:
            # Menulis byte bobot ke file temporary.
            temp_file.write(weights_bytes)

            # Memastikan seluruh byte tersimpan.
            temp_file.flush()

        # Menutup file sementara setelah penulisan.
        finally:
            temp_file.close()

        # Mengembalikan path file bobot sementara.
        return Path(temp_file.name)

    # Fungsi untuk memasangkan bobot kompatibel ke layer-layer model.
    def _assign_compat_weights(self, model, weights_path: Path) -> None:
        # h5py digunakan untuk membaca file bobot HDF5.
        import h5py  # type: ignore

        # Daftar path dataset bobot sesuai urutan layer arsitektur kompatibel.
        dataset_paths = [
            "layers/conv_lstm2d/cell/vars/0",
            "layers/conv_lstm2d/cell/vars/1",
            "layers/conv_lstm2d/cell/vars/2",
            "layers/batch_normalization/vars/0",
            "layers/batch_normalization/vars/1",
            "layers/batch_normalization/vars/2",
            "layers/batch_normalization/vars/3",
            "layers/conv_lstm2d_1/cell/vars/0",
            "layers/conv_lstm2d_1/cell/vars/1",
            "layers/conv_lstm2d_1/cell/vars/2",
            "layers/batch_normalization_1/vars/0",
            "layers/batch_normalization_1/vars/1",
            "layers/batch_normalization_1/vars/2",
            "layers/batch_normalization_1/vars/3",
            "layers/conv2d/vars/0",
            "layers/conv2d/vars/1",
            "layers/conv2d_1/vars/0",
            "layers/conv2d_1/vars/1",
        ]

        # Membuka file bobot HDF5.
        with h5py.File(weights_path, "r") as handle:
            # Membaca seluruh array bobot berdasarkan daftar path dataset.
            arrays = [np.array(handle[path]) for path in dataset_paths]

        # Memasang bobot ConvLSTM pertama.
        model.layers[0].set_weights(arrays[0:3])

        # Memasang bobot BatchNormalization pertama.
        model.layers[1].set_weights(arrays[3:7])

        # Memasang bobot ConvLSTM kedua.
        model.layers[3].set_weights(arrays[7:10])

        # Memasang bobot BatchNormalization kedua.
        model.layers[4].set_weights(arrays[10:14])

        # Memasang bobot Conv2D pertama.
        model.layers[6].set_weights(arrays[14:16])

        # Memasang bobot Conv2D output.
        model.layers[7].set_weights(arrays[16:18])

    # Fungsi untuk mengubah sequence input menjadi grayscale 2D per frame.
    def _to_grayscale(self, sequence: np.ndarray) -> np.ndarray:
        # Mengambil data frame dari batch pertama.
        frames = sequence[0]

        # Jika channel terakhir sudah 1, ambil channel tunggal tersebut.
        if frames.shape[-1] == 1:
            return frames[:, :, :, 0]

        # Jika channel lebih dari 1, hitung rata-rata channel sebagai grayscale.
        return np.mean(frames, axis=-1)

    # Fungsi smoothing sederhana untuk menghaluskan probability map fallback.
    def _smooth_map(self, arr: np.ndarray) -> np.ndarray:
        # Menambahkan padding tepi agar operasi tetangga tidak keluar batas.
        padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")

        # Mengambil nilai piksel pusat.
        center = padded[1:-1, 1:-1]

        # Mengambil tetangga utara.
        north = padded[:-2, 1:-1]

        # Mengambil tetangga selatan.
        south = padded[2:, 1:-1]

        # Mengambil tetangga barat.
        west = padded[1:-1, :-2]

        # Mengambil tetangga timur.
        east = padded[1:-1, 2:]

        # Menggabungkan pusat dan tetangga untuk menghasilkan peta yang lebih halus.
        return (0.5 * center) + (0.125 * (north + south + west + east))

    # Fungsi prediksi fallback jika model ConvLSTM tidak tersedia pada mode direct_resize.
    def _predict_once_heuristic(self, sequence: np.ndarray) -> np.ndarray:
        # Mengubah sequence input menjadi grayscale.
        grayscale = self._to_grayscale(sequence)

        # Mengambil jumlah frame waktu.
        steps = grayscale.shape[0]

        # Membuat bobot temporal yang makin besar untuk frame terbaru.
        weights = np.linspace(1.0, float(steps), num=steps, dtype=np.float32)

        # Menormalisasi bobot agar jumlahnya 1.
        weights /= np.sum(weights)

        # Menghitung rata-rata historis berbobot dari seluruh frame.
        weighted_history = np.tensordot(weights, grayscale, axes=(0, 0))

        # Mengambil frame terakhir H0.
        last_frame = grayscale[-1]

        # Mengambil frame sebelumnya H-1.
        previous_frame = grayscale[-2]

        # Menghitung tren kenaikan hotspot dari H-1 ke H0.
        short_trend = np.clip(last_frame - previous_frame, 0.0, 1.0)

        # Menggabungkan frame terakhir, histori berbobot, dan tren pendek menjadi probabilitas mentah.
        raw_probability = (
            (0.55 * last_frame)
            + (0.30 * weighted_history)
            + (0.15 * short_trend)
        )

        # Menghaluskan probability map fallback.
        smoothed = self._smooth_map(np.clip(raw_probability, 0.0, 1.0))

        # Mengembalikan probability map dengan rentang 0 sampai 1.
        return np.clip(smoothed, 0.0, 1.0).astype(np.float32)

    # Fungsi prediksi satu langkah menggunakan model ConvLSTM pada mode direct_resize.
    def _predict_once_model(self, sequence: np.ndarray) -> np.ndarray:
        # Memastikan model sudah termuat.
        if self.model is None:
            # Jika belum termuat, proses model tidak dapat dijalankan.
            raise ValueError("Model ConvLSTM belum termuat.")

        # Menjalankan prediksi TensorFlow tanpa log verbose.
        raw = self.model.predict(sequence, verbose=0)

        # Mengubah hasil prediksi menjadi array float32.
        prediction = np.asarray(raw, dtype=np.float32)

        # Jika output 5D, buang dimensi batch pertama.
        if prediction.ndim == 5:
            prediction = prediction[0]

        # Jika output 4D dengan batch 1, buang dimensi batch.
        if prediction.ndim == 4 and prediction.shape[0] == 1:
            prediction = prediction[0]

        # Jika output 3D dengan channel terakhir 1, buang dimensi channel.
        if prediction.ndim == 3 and prediction.shape[-1] == 1:
            prediction = prediction[:, :, 0]

        # Jika output 3D dengan dimensi pertama 1, buang dimensi pertama.
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]

        # Setelah normalisasi shape, output harus berupa peta 2D.
        if prediction.ndim != 2:
            # Menghentikan proses jika output model tidak sesuai format probability map.
            raise ValueError(f"Output model tidak didukung: {prediction.shape}")

        # Membatasi output pada rentang 0 sampai 1.
        return np.clip(prediction, 0.0, 1.0).astype(np.float32)

    # Fungsi prediksi batch patch menggunakan model ConvLSTM.
    def _predict_batch_model(self, batch: np.ndarray) -> np.ndarray:
        # Memastikan model sudah termuat.
        if self.model is None:
            # Jika model tidak ada, batch model tidak dapat dijalankan.
            raise ValueError("Model ConvLSTM belum termuat.")

        # Menjalankan prediksi batch patch.
        raw = self.model.predict(batch, verbose=0)

        # Mengubah output menjadi array float32.
        prediction = np.asarray(raw, dtype=np.float32)

        # Jika output 5D dengan channel 1, ambil peta 2D tiap patch.
        if prediction.ndim == 5 and prediction.shape[-1] == 1:
            prediction = prediction[:, 0, :, :, 0]

        # Jika output 4D dengan channel 1, buang dimensi channel.
        elif prediction.ndim == 4 and prediction.shape[-1] == 1:
            prediction = prediction[:, :, :, 0]

        # Jika output bukan 3D setelah normalisasi, format tidak didukung.
        elif prediction.ndim != 3:
            raise ValueError(f"Output batch model tidak didukung: {prediction.shape}")

        # Mengembalikan batch probability map dengan rentang 0 sampai 1.
        return np.clip(prediction, 0.0, 1.0).astype(np.float32)

    # Fungsi untuk menentukan posisi awal patch pada satu dimensi citra.
    def _patch_positions(self, length: int, patch_size: int, stride: int) -> list[int]:
        # Jika panjang citra lebih kecil atau sama dengan patch, cukup satu patch dari posisi 0.
        if length <= patch_size:
            return [0]

        # Membuat daftar posisi patch berdasarkan stride.
        positions = list(range(0, max(1, length - patch_size + 1), stride))

        # Menentukan posisi terakhir agar bagian ujung citra tetap tercakup.
        last = length - patch_size

        # Jika posisi terakhir belum tercakup, tambahkan posisi terakhir.
        if positions[-1] != last:
            positions.append(last)

        # Mengembalikan daftar posisi patch.
        return positions

    # Fungsi untuk membentuk batch patch dari sequence mask ukuran penuh.
    def _build_patch_batch(
        self,
        # Sequence input dengan shape time, height, width.
        sequence: np.ndarray,
        # Ukuran patch yang akan dipotong.
        patch_size: int,
        # Jarak pergeseran antarpatch.
        stride: int,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        # Mengambil tinggi dan lebar dari sequence.
        _, height, width = sequence.shape

        # Menentukan posisi patch pada sumbu Y.
        y_positions = self._patch_positions(height, patch_size, stride)

        # Menentukan posisi patch pada sumbu X.
        x_positions = self._patch_positions(width, patch_size, stride)

        # Menghitung tinggi setelah padding jika patch terakhir melewati batas citra.
        padded_height = max(height, y_positions[-1] + patch_size)

        # Menghitung lebar setelah padding jika patch terakhir melewati batas citra.
        padded_width = max(width, x_positions[-1] + patch_size)

        # Menghitung jumlah padding tinggi.
        pad_height = padded_height - height

        # Menghitung jumlah padding lebar.
        pad_width = padded_width - width

        # Memberi padding nol pada sequence agar patch di tepi tetap berukuran sama.
        padded = np.pad(sequence, ((0, 0), (0, pad_height), (0, pad_width)), mode="constant")

        # Menyiapkan list patch.
        patches: list[np.ndarray] = []

        # Menyiapkan list koordinat patch.
        coords: list[tuple[int, int]] = []

        # Iterasi posisi patch pada sumbu Y.
        for y in y_positions:
            # Iterasi posisi patch pada sumbu X.
            for x in x_positions:
                # Mengambil potongan patch dari sequence.
                patch = padded[:, y : y + patch_size, x : x + patch_size]

                # Jika model memakai 1 channel, tambahkan dimensi channel.
                if self.spec.channels == 1:
                    patch = np.expand_dims(patch, axis=-1)

                # Jika model memakai lebih dari 1 channel, duplikasi mask ke semua channel.
                else:
                    patch = np.repeat(np.expand_dims(patch, axis=-1), repeats=self.spec.channels, axis=-1)

                # Menyimpan patch ke list batch.
                patches.append(patch)

                # Menyimpan koordinat kiri-atas patch.
                coords.append((y, x))

        # Mengembalikan batch patch dan koordinatnya.
        return np.stack(patches, axis=0).astype(np.float32), coords

    # Fungsi untuk memprediksi seluruh patch dalam beberapa batch kecil.
    def _predict_patch_batch(self, batch: np.ndarray) -> np.ndarray:
        # Jika model tidak tersedia, gunakan fallback berbasis frame terakhir dan rata-rata historis.
        if self.model is None:
            # Mengambil frame terakhir pada setiap patch.
            last_frame = batch[:, -1, :, :, 0]

            # Menghitung rata-rata historis semua frame pada setiap patch.
            weighted = np.mean(batch[:, :, :, :, 0], axis=1)

            # Menggabungkan frame terakhir dan rata-rata historis sebagai probability fallback.
            return np.clip((0.7 * last_frame) + (0.3 * weighted), 0.0, 1.0).astype(np.float32)

        # Menentukan ukuran batch patch sesuai konfigurasi, default 32.
        batch_size = self.spec.patch_batch_size or 32

        # Menyiapkan list hasil prediksi patch.
        predictions: list[np.ndarray] = []

        # Memproses patch secara bertahap agar penggunaan memori lebih aman.
        for start in range(0, batch.shape[0], batch_size):
            # Mengambil potongan batch patch.
            chunk = batch[start : start + batch_size]

            # Menjalankan prediksi model pada chunk patch.
            predictions.append(self._predict_batch_model(chunk))

        # Menggabungkan seluruh hasil prediksi patch.
        return np.concatenate(predictions, axis=0)

    # Fungsi untuk menggabungkan hasil prediksi patch menjadi peta utuh.
    def _stitch_patch_predictions(
        self,
        # Hasil prediksi setiap patch.
        predictions: np.ndarray,
        # Koordinat kiri-atas setiap patch.
        coords: list[tuple[int, int]],
        # Ukuran output asli sebelum padding.
        output_shape: tuple[int, int],
        # Ukuran patch.
        patch_size: int,
    ) -> np.ndarray:
        # Mengambil tinggi dan lebar output asli.
        height, width = output_shape

        # Menghitung tinggi canvas setelah padding.
        padded_height = max(height, max(y for y, _ in coords) + patch_size)

        # Menghitung lebar canvas setelah padding.
        padded_width = max(width, max(x for _, x in coords) + patch_size)

        # Canvas menyimpan akumulasi nilai prediksi patch.
        canvas = np.zeros((padded_height, padded_width), dtype=np.float32)

        # Weights menyimpan jumlah patch yang menutupi tiap piksel.
        weights = np.zeros((padded_height, padded_width), dtype=np.float32)

        # Menggabungkan setiap patch prediksi ke canvas berdasarkan koordinatnya.
        for prediction, (y, x) in zip(predictions, coords):
            # Menambahkan nilai prediksi patch ke area canvas.
            canvas[y : y + patch_size, x : x + patch_size] += prediction

            # Menambahkan bobot 1 pada area yang ditutupi patch.
            weights[y : y + patch_size, x : x + patch_size] += 1.0

        # Membagi akumulasi prediksi dengan bobot agar overlap menjadi rata-rata.
        stitched = canvas / np.maximum(weights, 1e-6)

        # Memotong kembali hasil sesuai ukuran asli dan membatasi nilai 0 sampai 1.
        return np.clip(stitched[:height, :width], 0.0, 1.0).astype(np.float32)

    # Fungsi prediksi satu langkah menggunakan mode patch_stitch.
    def _predict_once_patch_stitch(self, sequence: np.ndarray) -> np.ndarray:
        # Sequence patch harus berbentuk 3D: time, height, width.
        if sequence.ndim != 3:
            # Menghentikan proses jika shape sequence tidak sesuai.
            raise ValueError(f"Sequence patch harus 3D (time, height, width), dapat {sequence.shape}")

        # Mengambil ukuran patch dari spesifikasi; jika tidak ada, gunakan grid_size.
        patch_size = self.spec.patch_size or self.spec.grid_size

        # Mengambil stride dari spesifikasi; jika tidak ada, gunakan patch_size.
        stride = self.spec.patch_stride or patch_size

        # Mengambil tinggi dan lebar sequence.
        _, height, width = sequence.shape

        # Membentuk batch patch dan koordinat patch.
        batch, coords = self._build_patch_batch(sequence=sequence, patch_size=patch_size, stride=stride)

        # Mencoba menjalankan prediksi patch.
        try:
            # Prediksi patch memakai model jika tersedia atau fallback jika model None.
            predictions = self._predict_patch_batch(batch)

        # Jika terjadi error runtime, sistem jatuh ke fallback heuristic.
        except Exception:
            # Mengosongkan model karena terjadi kegagalan runtime.
            self.model = None

            # Mengosongkan path model aktif.
            self.model_path = None

            # Menandai backend sebagai runtime fallback.
            self.backend = "heuristic-temporal-v1 (runtime-fallback)"

            # Menjalankan ulang prediksi patch dengan fallback.
            predictions = self._predict_patch_batch(batch)

        # Menggabungkan kembali prediksi patch menjadi probability map utuh.
        return self._stitch_patch_predictions(
            predictions=predictions,
            coords=coords,
            output_shape=(height, width),
            patch_size=patch_size,
        )

    # Fungsi prediksi beberapa horizon pada mode patch_stitch.
    def _predict_horizon_patch_stitch(self, sequence: np.ndarray, horizon: int) -> list[np.ndarray]:
        # Menyalin sequence agar input asli tidak berubah.
        running = np.copy(sequence)

        # Menyiapkan list output prediksi.
        outputs: list[np.ndarray] = []

        # Melakukan prediksi sebanyak horizon.
        for _ in range(horizon):
            # Memprediksi satu langkah berikutnya.
            predicted = self._predict_once_patch_stitch(running)

            # Menyimpan hasil prediksi ke list output.
            outputs.append(predicted)

            # Menggeser sequence: frame terlama dibuang dan prediksi menjadi frame terbaru.
            running = np.concatenate([running[1:], np.expand_dims(predicted, axis=0)], axis=0)

        # Mengembalikan daftar probability map hasil prediksi.
        return outputs

    # Fungsi publik untuk menjalankan prediksi sesuai mode inferensi aktif.
    def predict_horizon(self, sequence: np.ndarray, horizon: int = 1) -> list[np.ndarray]:
        # Horizon minimal harus 1.
        if horizon < 1:
            # Menghentikan proses jika horizon tidak valid.
            raise ValueError("Horizon minimal adalah 1.")

        # Jika mode inferensi patch_stitch, gunakan pipeline patch.
        if self.spec.inference_mode == "patch_stitch":
            # Mengembalikan hasil prediksi patch_stitch.
            return self._predict_horizon_patch_stitch(sequence=sequence, horizon=horizon)

        # Menyalin sequence untuk mode direct_resize.
        running = np.copy(sequence)

        # Menyiapkan list output prediksi.
        outputs: list[np.ndarray] = []

        # Melakukan prediksi sebanyak horizon.
        for _ in range(horizon):
            # Mencoba prediksi memakai model jika tersedia.
            try:
                # Jika model ada, gunakan model ConvLSTM.
                if self.model is not None:
                    predicted = self._predict_once_model(running)

                # Jika model tidak ada, gunakan fallback heuristic.
                else:
                    predicted = self._predict_once_heuristic(running)

            # Jika terjadi error runtime model, sistem otomatis fallback.
            except Exception:
                # Mengosongkan model karena gagal saat runtime.
                self.model = None

                # Mengosongkan path model aktif.
                self.model_path = None

                # Menandai backend sebagai runtime fallback.
                self.backend = "heuristic-temporal-v1 (runtime-fallback)"

                # Menjalankan prediksi heuristic sebagai pengganti model.
                predicted = self._predict_once_heuristic(running)

            # Menyimpan hasil prediksi.
            outputs.append(predicted)

            # Menambahkan dimensi channel pada output prediksi agar bisa masuk sequence berikutnya.
            next_frame = np.expand_dims(predicted, axis=-1)

            # Jika model membutuhkan 3 channel, prediksi diduplikasi ke 3 channel.
            if self.spec.channels == 3:
                next_frame = np.repeat(next_frame, repeats=3, axis=-1)

            # Menambahkan dimensi batch/time kecil untuk penyusunan ulang sequence.
            next_frame = np.expand_dims(next_frame, axis=0)

            # Mengambil sequence tanpa dimensi batch.
            without_batch = running[0]

            # Menggeser sequence: frame lama dibuang, prediksi baru masuk sebagai frame terakhir.
            shifted = np.concatenate([without_batch[1:], next_frame], axis=0)

            # Menambahkan kembali dimensi batch.
            running = np.expand_dims(shifted, axis=0)

        # Mengembalikan daftar hasil prediksi.
        return outputs
