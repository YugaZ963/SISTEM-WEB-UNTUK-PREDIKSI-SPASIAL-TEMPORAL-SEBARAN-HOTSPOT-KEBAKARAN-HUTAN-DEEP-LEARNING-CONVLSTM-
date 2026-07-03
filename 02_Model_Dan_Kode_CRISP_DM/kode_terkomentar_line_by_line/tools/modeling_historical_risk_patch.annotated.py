# File anotasi dari `tools/modeling_historical_risk_patch.py`.
# File ini untuk belajar/lampiran, bukan source utama aplikasi.
# Konteks laporan: Modeling, yaitu membangun dan melatih model ConvLSTM.
# Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
# Catatan pembuka/penutup yang menjelaskan isi bagian kode.
"""Komentar file skripsi:
# Isi catatan penjelasan pada bagian kode ini.
Script ringkas fase Modeling CRISP-DM untuk menjelaskan arsitektur ConvLSTM historical_risk_patch_160.

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
import contextlib
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import io
# Mengambil alat bantu/library yang diperlukan oleh file ini.
import json


# Membuat langkah kerja bernama `feature_stack_channels`.
def feature_stack_channels(feature_stack: str) -> int:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return 3 if feature_stack == "mask_context3" else 1


# Membuat langkah kerja bernama `build_parser`.
def build_parser() -> argparse.ArgumentParser:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = argparse.ArgumentParser(
        # Menyimpan nilai ke `description` untuk dipakai pada langkah berikutnya.
        description=(
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Ringkasan fase Modeling untuk historical risk patch. "
            # Melanjutkan langkah kerja pada bagian kode ini.
            "Script ini fokus pada arsitektur model, loss, optimizer, dan metrics tanpa menjalankan training."
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--seq-length", type=int, default=7)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--patch-width", type=int, default=160)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--patch-height", type=int, default=160)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--feature-stack", choices=["mask", "mask_context3"], default="mask")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--loss-strategy", choices=["wbce_dice", "wbce_dice_context"], default="wbce_dice_context")
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--context-kernel", type=int, default=5)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--pos-weight", type=float, default=50.0)
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--skip-tensorflow",
        # Menyimpan nilai ke `action` untuk dipakai pada langkah berikutnya.
        action="store_true",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Jangan instantiate model TensorFlow. Cocok jika hanya ingin ringkasan statis untuk screenshot kode.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument(
        # Melanjutkan langkah kerja pada bagian kode ini.
        "--save-summary",
        # Menyimpan nilai ke `default` untuk dipakai pada langkah berikutnya.
        default="",
        # Menyimpan nilai ke `help` untuk dipakai pada langkah berikutnya.
        help="Path file txt untuk menyimpan model.summary(). Diabaikan jika TensorFlow tidak tersedia.",
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menambahkan pilihan saat script dijalankan dari terminal.
    parser.add_argument("--json", action="store_true")
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return parser


# Membuat langkah kerja bernama `build_static_layer_specs`.
def build_static_layer_specs(channels: int, seq_length: int, patch_height: int, patch_width: int) -> list[dict]:
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return [
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "Input",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "Input",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "shape": [seq_length, patch_height, patch_width, channels],
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "ConvLSTM Block 1",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "ConvLSTM2D",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "filters": 32,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "kernel_size": [3, 3],
            # Melanjutkan langkah kerja pada bagian kode ini.
            "padding": "same",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "return_sequences": True,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "Normalization 1",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "BatchNormalization",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "Dropout 1",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "Dropout",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "rate": 0.2,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "ConvLSTM Block 2",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "ConvLSTM2D",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "filters": 32,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "kernel_size": [3, 3],
            # Melanjutkan langkah kerja pada bagian kode ini.
            "padding": "same",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "return_sequences": False,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "Normalization 2",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "BatchNormalization",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "Dropout 2",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "Dropout",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "rate": 0.2,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "Conv Head",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "Conv2D",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "filters": 16,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "kernel_size": [3, 3],
            # Melanjutkan langkah kerja pada bagian kode ini.
            "activation": "relu",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "padding": "same",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            # Melanjutkan langkah kerja pada bagian kode ini.
            "name": "Output Head",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "type": "Conv2D",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "filters": 1,
            # Melanjutkan langkah kerja pada bagian kode ini.
            "kernel_size": [1, 1],
            # Melanjutkan langkah kerja pada bagian kode ini.
            "activation": "sigmoid",
            # Melanjutkan langkah kerja pada bagian kode ini.
            "padding": "same",
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        },
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    ]


# Membuat langkah kerja bernama `import_tensorflow`.
def import_tensorflow():
    # Mencoba menjalankan proses yang mungkin gagal.
    try:
        # Mengambil alat bantu/library yang diperlukan oleh file ini.
        import tensorflow as tf  # type: ignore

        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return tf, None
    # Menangani kesalahan agar program tidak langsung berhenti.
    except Exception as exc:
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return None, str(exc)


# Membuat langkah kerja bernama `_compute_weighted_bce`.
def _compute_weighted_bce(y_true, y_pred, pos_weight_tensor, tf):
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tf.reduce_mean(
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        -(
            # Melanjutkan langkah kerja pada bagian kode ini.
            (pos_weight_tensor * y_true * tf.math.log(y_pred))
            # Melanjutkan langkah kerja pada bagian kode ini.
            + ((1.0 - y_true) * tf.math.log(1.0 - y_pred))
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        )
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )


# Membuat langkah kerja bernama `_compute_dice_loss`.
def _compute_dice_loss(y_true, y_pred, tf):
    # Menyimpan nilai ke `intersection` untuk dipakai pada langkah berikutnya.
    intersection = tf.reduce_sum(y_true * y_pred)
    # Menyimpan nilai ke `denominator` untuk dipakai pada langkah berikutnya.
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))


# Membuat langkah kerja bernama `_compute_context_bce`.
def _compute_context_bce(y_true, y_pred, context_kernel: int, tf):
    # Menyimpan nilai ke `pooled_context` untuk dipakai pada langkah berikutnya.
    pooled_context = tf.nn.max_pool2d(y_true, ksize=context_kernel, strides=1, padding="SAME")
    # Menyimpan nilai ke `context_weight` untuk dipakai pada langkah berikutnya.
    context_weight = 1.0 + (2.0 * pooled_context)
    # Menyimpan nilai ke `context_bce_map` untuk dipakai pada langkah berikutnya.
    context_bce_map = -(
        # Melanjutkan langkah kerja pada bagian kode ini.
        (y_true * tf.math.log(y_pred))
        # Melanjutkan langkah kerja pada bagian kode ini.
        + ((1.0 - y_true) * tf.math.log(1.0 - y_pred))
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return tf.reduce_mean(context_weight * context_bce_map)


# Membuat langkah kerja bernama `make_weighted_bce_dice_loss`.
def make_weighted_bce_dice_loss(pos_weight: float, tf):
    # Menyimpan nilai ke `pos_weight_tensor` untuk dipakai pada langkah berikutnya.
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    # Menyimpan nilai ke `epsilon` untuk dipakai pada langkah berikutnya.
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Membuat langkah kerja bernama `loss`.
    def loss(y_true, y_pred):
        # Menyimpan nilai ke `y_true` untuk dipakai pada langkah berikutnya.
        y_true = tf.cast(y_true, tf.float32)
        # Menyimpan nilai ke `y_pred` untuk dipakai pada langkah berikutnya.
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        # Menyimpan nilai ke `weighted_bce` untuk dipakai pada langkah berikutnya.
        weighted_bce = _compute_weighted_bce(y_true, y_pred, pos_weight_tensor, tf)
        # Menyimpan nilai ke `dice_loss` untuk dipakai pada langkah berikutnya.
        dice_loss = _compute_dice_loss(y_true, y_pred, tf)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return (0.7 * weighted_bce) + (0.3 * dice_loss)

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return loss


# Membuat langkah kerja bernama `make_weighted_bce_dice_context_loss`.
def make_weighted_bce_dice_context_loss(pos_weight: float, context_kernel: int, tf):
    # Menyimpan nilai ke `pos_weight_tensor` untuk dipakai pada langkah berikutnya.
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    # Menyimpan nilai ke `epsilon` untuk dipakai pada langkah berikutnya.
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Membuat langkah kerja bernama `loss`.
    def loss(y_true, y_pred):
        # Menyimpan nilai ke `y_true` untuk dipakai pada langkah berikutnya.
        y_true = tf.cast(y_true, tf.float32)
        # Menyimpan nilai ke `y_pred` untuk dipakai pada langkah berikutnya.
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        # Menyimpan nilai ke `weighted_bce` untuk dipakai pada langkah berikutnya.
        weighted_bce = _compute_weighted_bce(y_true, y_pred, pos_weight_tensor, tf)
        # Menyimpan nilai ke `dice_loss` untuk dipakai pada langkah berikutnya.
        dice_loss = _compute_dice_loss(y_true, y_pred, tf)
        # Menyimpan nilai ke `context_bce` untuk dipakai pada langkah berikutnya.
        context_bce = _compute_context_bce(y_true, y_pred, context_kernel, tf)
        # Menyimpan nilai ke `base_loss` untuk dipakai pada langkah berikutnya.
        base_loss = (0.7 * weighted_bce) + (0.3 * dice_loss)
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return (0.75 * base_loss) + (0.25 * context_bce)

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return loss


# Membuat langkah kerja bernama `build_loss`.
def build_loss(loss_strategy: str, pos_weight: float, context_kernel: int, tf):
    # Mengecek syarat sebelum melanjutkan proses.
    if loss_strategy == "wbce_dice_context":
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return make_weighted_bce_dice_context_loss(pos_weight, context_kernel, tf)
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return make_weighted_bce_dice_loss(pos_weight, tf)


# Membuat langkah kerja bernama `build_model`.
def build_model(
    # Melanjutkan langkah kerja pada bagian kode ini.
    *,
    # Menjelaskan data `seq_length` yang disimpan atau dikirim pada bagian ini.
    seq_length: int,
    # Menjelaskan data `patch_height` yang disimpan atau dikirim pada bagian ini.
    patch_height: int,
    # Menjelaskan data `patch_width` yang disimpan atau dikirim pada bagian ini.
    patch_width: int,
    # Menjelaskan data `channels` yang disimpan atau dikirim pada bagian ini.
    channels: int,
    # Menjelaskan data `pos_weight` yang disimpan atau dikirim pada bagian ini.
    pos_weight: float,
    # Menjelaskan data `loss_strategy` yang disimpan atau dikirim pada bagian ini.
    loss_strategy: str,
    # Menjelaskan data `context_kernel` yang disimpan atau dikirim pada bagian ini.
    context_kernel: int,
    # Menyebutkan item yang ikut dipakai pada daftar di atas.
    tf,
# Menutup susunan data atau perintah yang dimulai sebelumnya.
):
    # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
    model = tf.keras.Sequential(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            # Menyimpan nilai ke `tf.keras.layers.Input(shape` untuk dipakai pada langkah berikutnya.
            tf.keras.layers.Input(shape=(seq_length, patch_height, patch_width, channels)),
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
    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    model.compile(
        # Menyimpan nilai ke `optimizer` untuk dipakai pada langkah berikutnya.
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # Menyimpan nilai ke `loss` untuk dipakai pada langkah berikutnya.
        loss=build_loss(loss_strategy, pos_weight, context_kernel, tf),
        # Menyimpan nilai ke `jit_compile` untuk dipakai pada langkah berikutnya.
        jit_compile=False,
        # Menyimpan nilai ke `metrics` untuk dipakai pada langkah berikutnya.
        metrics=[
            # Menyimpan nilai ke `tf.keras.metrics.BinaryAccuracy(name` untuk dipakai pada langkah berikutnya.
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            # Menyimpan nilai ke `tf.keras.metrics.Precision(name` untuk dipakai pada langkah berikutnya.
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            # Menyimpan nilai ke `tf.keras.metrics.Recall(name` untuk dipakai pada langkah berikutnya.
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
            # Menyimpan nilai ke `tf.keras.metrics.AUC(name` untuk dipakai pada langkah berikutnya.
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ],
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return model


# Membuat langkah kerja bernama `capture_model_summary`.
def capture_model_summary(model) -> str:
    # Menyimpan nilai ke `buffer` untuk dipakai pada langkah berikutnya.
    buffer = io.StringIO()
    # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
    with contextlib.redirect_stdout(buffer):
        # Melanjutkan langkah kerja pada bagian kode ini.
        model.summary()
    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return buffer.getvalue().strip()


# Membuat langkah kerja bernama `build_summary`.
def build_summary(args: argparse.Namespace) -> dict:
    # Menyimpan nilai ke `channels` untuk dipakai pada langkah berikutnya.
    channels = feature_stack_channels(args.feature_stack)
    # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
    summary: dict = {
        # Melanjutkan langkah kerja pada bagian kode ini.
        "phase": "modeling",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_profile": "historical_risk_patch_160",
        # Melanjutkan langkah kerja pada bagian kode ini.
        "seq_length": args.seq_length,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_width": args.patch_width,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "patch_height": args.patch_height,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "feature_stack": args.feature_stack,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "channels": channels,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "loss_strategy": args.loss_strategy,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "context_kernel": args.context_kernel,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "positive_class_weight": args.pos_weight,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "optimizer": {"name": "Adam", "learning_rate": 0.001},
        # Melanjutkan langkah kerja pada bagian kode ini.
        "metrics": ["accuracy", "precision", "recall", "pr_auc"],
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "static_layers": build_static_layer_specs(
            # Menyimpan nilai ke `channels` untuk dipakai pada langkah berikutnya.
            channels=channels,
            # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
            seq_length=args.seq_length,
            # Menyimpan nilai ke `patch_height` untuk dipakai pada langkah berikutnya.
            patch_height=args.patch_height,
            # Menyimpan nilai ke `patch_width` untuk dipakai pada langkah berikutnya.
            patch_width=args.patch_width,
        # Menutup susunan data atau perintah yang dimulai sebelumnya.
        ),
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tensorflow_available": False,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "tensorflow_error": None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_instantiated": False,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "parameter_count": None,
        # Melanjutkan langkah kerja pada bagian kode ini.
        "model_summary_text": None,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    }

    # Mengecek syarat sebelum melanjutkan proses.
    if args.skip_tensorflow:
        # Menyimpan nilai ke `summary["tensorflow_error"]` untuk dipakai pada langkah berikutnya.
        summary["tensorflow_error"] = "skip_tensorflow=True"
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return summary

    # Menyimpan nilai ke `tf, tf_error` untuk dipakai pada langkah berikutnya.
    tf, tf_error = import_tensorflow()
    # Mengecek syarat sebelum melanjutkan proses.
    if tf is None:
        # Menyimpan nilai ke `summary["tensorflow_error"]` untuk dipakai pada langkah berikutnya.
        summary["tensorflow_error"] = tf_error
        # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
        return summary

    # Menyimpan nilai ke `summary["tensorflow_available"]` untuk dipakai pada langkah berikutnya.
    summary["tensorflow_available"] = True
    # Menyimpan nilai ke `model` untuk dipakai pada langkah berikutnya.
    model = build_model(
        # Menyimpan nilai ke `seq_length` untuk dipakai pada langkah berikutnya.
        seq_length=args.seq_length,
        # Menyimpan nilai ke `patch_height` untuk dipakai pada langkah berikutnya.
        patch_height=args.patch_height,
        # Menyimpan nilai ke `patch_width` untuk dipakai pada langkah berikutnya.
        patch_width=args.patch_width,
        # Menyimpan nilai ke `channels` untuk dipakai pada langkah berikutnya.
        channels=channels,
        # Menyimpan nilai ke `pos_weight` untuk dipakai pada langkah berikutnya.
        pos_weight=args.pos_weight,
        # Menyimpan nilai ke `loss_strategy` untuk dipakai pada langkah berikutnya.
        loss_strategy=args.loss_strategy,
        # Menyimpan nilai ke `context_kernel` untuk dipakai pada langkah berikutnya.
        context_kernel=args.context_kernel,
        # Menyimpan nilai ke `tf` untuk dipakai pada langkah berikutnya.
        tf=tf,
    # Menutup susunan data atau perintah yang dimulai sebelumnya.
    )
    # Menyimpan nilai ke `summary["model_instantiated"]` untuk dipakai pada langkah berikutnya.
    summary["model_instantiated"] = True
    # Menyimpan nilai ke `summary["parameter_count"]` untuk dipakai pada langkah berikutnya.
    summary["parameter_count"] = int(model.count_params())
    # Menyimpan nilai ke `summary["model_summary_text"]` untuk dipakai pada langkah berikutnya.
    summary["model_summary_text"] = capture_model_summary(model)

    # Mengecek syarat sebelum melanjutkan proses.
    if args.save_summary:
        # Membuka file/gambar dengan aman lalu menutupnya otomatis setelah selesai.
        with open(args.save_summary, "w", encoding="utf-8") as handle:
            # Melanjutkan langkah kerja pada bagian kode ini.
            handle.write(str(summary["model_summary_text"]))
            # Melanjutkan langkah kerja pada bagian kode ini.
            handle.write("\n")
        # Menyimpan nilai ke `summary["model_summary_saved_to"]` untuk dipakai pada langkah berikutnya.
        summary["model_summary_saved_to"] = args.save_summary

    # Mengirim hasil dari langkah kerja ini ke bagian yang memanggilnya.
    return summary


# Membuat langkah kerja bernama `print_human_summary`.
def print_human_summary(summary: dict) -> None:
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Profile model:", summary["model_profile"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Seq length:", summary["seq_length"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Patch size:", f"{summary['patch_width']}x{summary['patch_height']}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Feature stack:", summary["feature_stack"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Channels:", summary["channels"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Loss strategy:", summary["loss_strategy"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Context kernel:", summary["context_kernel"])
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Positive class weight:", f"{summary['positive_class_weight']:.4f}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Optimizer:", f"{summary['optimizer']['name']} lr={summary['optimizer']['learning_rate']}")
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Metrics:", ", ".join(summary["metrics"]))
    # Menampilkan informasi ke terminal agar proses mudah dicek.
    print("[modeling] Layer utama:")
    # Mengulang proses untuk setiap data dalam daftar.
    for layer in summary["static_layers"]:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(f"- {layer['name']} | {layer['type']}")

    # Mengecek syarat sebelum melanjutkan proses.
    if summary["tensorflow_available"] and summary["model_instantiated"]:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[modeling] TensorFlow: tersedia")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[modeling] Parameter model:", summary["parameter_count"])
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print()
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print(summary["model_summary_text"])
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[modeling] TensorFlow: tidak tersedia atau dilewati")
        # Menampilkan informasi ke terminal agar proses mudah dicek.
        print("[modeling] Detail:", summary["tensorflow_error"])


# Membuat langkah kerja bernama `main`.
def main() -> None:
    # Menyimpan nilai ke `parser` untuk dipakai pada langkah berikutnya.
    parser = build_parser()
    # Menyimpan nilai ke `args` untuk dipakai pada langkah berikutnya.
    args = parser.parse_args()
    # Menyimpan nilai ke `summary` untuk dipakai pada langkah berikutnya.
    summary = build_summary(args)
    # Melanjutkan langkah kerja pada bagian kode ini.
    print_human_summary(summary)
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
