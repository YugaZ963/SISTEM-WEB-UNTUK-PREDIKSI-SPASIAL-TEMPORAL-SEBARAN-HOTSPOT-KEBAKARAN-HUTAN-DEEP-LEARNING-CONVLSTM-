
"""Komentar file skripsi:
Script ringkas fase Modeling CRISP-DM untuk menjelaskan arsitektur ConvLSTM historical_risk_patch_160.

Konteks laporan: file ini mendukung tahapan CRISP-DM pada BAB IV,
terutama data understanding, data preparation, modeling, evaluation,
atau deployment sesuai fungsi teknisnya.
"""

from __future__ import annotations

# argparse dipakai agar parameter dataset, patch, threshold, dan output bisa diatur dari command line.
import argparse
import contextlib
# io dipakai di test untuk membuat file gambar sementara di memori tanpa menulis file manual.
import io
# json dipakai untuk menyimpan ringkasan eksperimen, metrik, dan metadata proses.
import json


# Fungsi `feature_stack_channels` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def feature_stack_channels(feature_stack: str) -> int:
    # Hasil ini dikembalikan sebagai output fungsi `feature_stack_channels` untuk tahap berikutnya.
    return 3 if feature_stack == "mask_context3" else 1


# Menyusun opsi command line agar fase CRISP-DM bisa dijalankan ulang dengan parameter yang jelas.
def build_parser() -> argparse.ArgumentParser:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = argparse.ArgumentParser(
        description=(
            "Ringkasan fase Modeling untuk historical risk patch. "
            "Script ini fokus pada arsitektur model, loss, optimizer, dan metrics tanpa menjalankan training."
        )
    )
    # Opsi ini menentukan jumlah frame historis; project memakai 7 frame H-6 sampai H0.
    parser.add_argument("--seq-length", type=int, default=7)
    # Opsi ini menentukan lebar patch yang dipotong dari citra besar.
    parser.add_argument("--patch-width", type=int, default=160)
    # Opsi ini menentukan tinggi patch yang dipotong dari citra besar.
    parser.add_argument("--patch-height", type=int, default=160)
    # Opsi `--feature-stack` menambah parameter eksekusi script.
    parser.add_argument("--feature-stack", choices=["mask", "mask_context3"], default="mask")
    # Opsi `--loss-strategy` menambah parameter eksekusi script.
    parser.add_argument("--loss-strategy", choices=["wbce_dice", "wbce_dice_context"], default="wbce_dice_context")
    # Opsi `--context-kernel` menambah parameter eksekusi script.
    parser.add_argument("--context-kernel", type=int, default=5)
    # Opsi `--pos-weight` menambah parameter eksekusi script.
    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument(
        "--skip-tensorflow",
        action="store_true",
        help="Jangan instantiate model TensorFlow. Cocok jika hanya ingin ringkasan statis untuk screenshot kode.",
    )
    parser.add_argument(
        "--save-summary",
        default="",
        help="Path file txt untuk menyimpan model.summary(). Diabaikan jika TensorFlow tidak tersedia.",
    )
    # Opsi ini membuat output ringkasan dicetak sebagai JSON agar mudah dikutip/dibaca ulang.
    parser.add_argument("--json", action="store_true")
    # Hasil ini dikembalikan sebagai output fungsi `build_parser` untuk tahap berikutnya.
    return parser


# Fungsi `build_static_layer_specs` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def build_static_layer_specs(channels: int, seq_length: int, patch_height: int, patch_width: int) -> list[dict]:
    # Hasil ini dikembalikan sebagai output fungsi `build_static_layer_specs` untuk tahap berikutnya.
    return [
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "Input",
            "type": "Input",
            "shape": [seq_length, patch_height, patch_width, channels],
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "ConvLSTM Block 1",
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            "type": "ConvLSTM2D",
            "filters": 32,
            "kernel_size": [3, 3],
            "padding": "same",
            "return_sequences": True,
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "Normalization 1",
            "type": "BatchNormalization",
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "Dropout 1",
            "type": "Dropout",
            "rate": 0.2,
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "ConvLSTM Block 2",
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            "type": "ConvLSTM2D",
            "filters": 32,
            "kernel_size": [3, 3],
            "padding": "same",
            "return_sequences": False,
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "Normalization 2",
            "type": "BatchNormalization",
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "Dropout 2",
            "type": "Dropout",
            "rate": 0.2,
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "Conv Head",
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            "type": "Conv2D",
            "filters": 16,
            "kernel_size": [3, 3],
            "activation": "relu",
            "padding": "same",
        },
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        {
            "name": "Output Head",
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            "type": "Conv2D",
            "filters": 1,
            "kernel_size": [1, 1],
            "activation": "sigmoid",
            "padding": "same",
        },
    ]


# Fungsi `import_tensorflow` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore

        # Hasil ini dikembalikan sebagai output fungsi `import_tensorflow` untuk tahap berikutnya.
        return tf, None
    except Exception as exc:
        # Hasil ini dikembalikan sebagai output fungsi `import_tensorflow` untuk tahap berikutnya.
        return None, str(exc)


# Fungsi `_compute_weighted_bce` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _compute_weighted_bce(y_true, y_pred, pos_weight_tensor, tf):
    # Hasil ini dikembalikan sebagai output fungsi `_compute_weighted_bce` untuk tahap berikutnya.
    return tf.reduce_mean(
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        -(
            (pos_weight_tensor * y_true * tf.math.log(y_pred))
            + ((1.0 - y_true) * tf.math.log(1.0 - y_pred))
        )
    )


# Fungsi `_compute_dice_loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _compute_dice_loss(y_true, y_pred, tf):
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    # Hasil ini dikembalikan sebagai output fungsi `_compute_dice_loss` untuk tahap berikutnya.
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))


# Fungsi `_compute_context_bce` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def _compute_context_bce(y_true, y_pred, context_kernel: int, tf):
    pooled_context = tf.nn.max_pool2d(y_true, ksize=context_kernel, strides=1, padding="SAME")
    context_weight = 1.0 + (2.0 * pooled_context)
    context_bce_map = -(
        (y_true * tf.math.log(y_pred))
        + ((1.0 - y_true) * tf.math.log(1.0 - y_pred))
    )
    # Hasil ini dikembalikan sebagai output fungsi `_compute_context_bce` untuk tahap berikutnya.
    return tf.reduce_mean(context_weight * context_bce_map)


# Fungsi `make_weighted_bce_dice_loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def make_weighted_bce_dice_loss(pos_weight: float, tf):
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Fungsi `loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        weighted_bce = _compute_weighted_bce(y_true, y_pred, pos_weight_tensor, tf)
        dice_loss = _compute_dice_loss(y_true, y_pred, tf)
        # Hasil ini dikembalikan sebagai output fungsi `loss` untuk tahap berikutnya.
        return (0.7 * weighted_bce) + (0.3 * dice_loss)

    # Hasil ini dikembalikan sebagai output fungsi `loss` untuk tahap berikutnya.
    return loss


# Fungsi `make_weighted_bce_dice_context_loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def make_weighted_bce_dice_context_loss(pos_weight: float, context_kernel: int, tf):
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Fungsi `loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        weighted_bce = _compute_weighted_bce(y_true, y_pred, pos_weight_tensor, tf)
        dice_loss = _compute_dice_loss(y_true, y_pred, tf)
        context_bce = _compute_context_bce(y_true, y_pred, context_kernel, tf)
        base_loss = (0.7 * weighted_bce) + (0.3 * dice_loss)
        # Hasil ini dikembalikan sebagai output fungsi `loss` untuk tahap berikutnya.
        return (0.75 * base_loss) + (0.25 * context_bce)

    # Hasil ini dikembalikan sebagai output fungsi `loss` untuk tahap berikutnya.
    return loss


# Fungsi `build_loss` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def build_loss(loss_strategy: str, pos_weight: float, context_kernel: int, tf):
    if loss_strategy == "wbce_dice_context":
        # Hasil ini dikembalikan sebagai output fungsi `build_loss` untuk tahap berikutnya.
        return make_weighted_bce_dice_context_loss(pos_weight, context_kernel, tf)
    # Hasil ini dikembalikan sebagai output fungsi `build_loss` untuk tahap berikutnya.
    return make_weighted_bce_dice_loss(pos_weight, tf)


# Menyusun arsitektur ConvLSTM yang menerima sequence 7 frame dan menghasilkan peta probabilitas H+1.
def build_model(
    *,
    seq_length: int,
    patch_height: int,
    patch_width: int,
    channels: int,
    pos_weight: float,
    loss_strategy: str,
    context_kernel: int,
    tf,
):
    # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
    model = tf.keras.Sequential(
        # Memulai susunan data atau perintah yang ditulis beberapa baris.
        [
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.Input(shape=(seq_length, patch_height, patch_width, channels)),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.BatchNormalization(),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.Dropout(0.2),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=False),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.BatchNormalization(),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.Dropout(0.2),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
            # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same"),
        ]
    )
    # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
    model.compile(
        # Baris ini menyusun komponen arsitektur model ConvLSTM untuk prediksi peta risiko.
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=build_loss(loss_strategy, pos_weight, context_kernel, tf),
        jit_compile=False,
        # Nilai precision, recall, F1, IoU, dan metrik lain hasil evaluasi.
        metrics=[
            # `tf.keras.metrics.BinaryAccuracy(name` menyimpan ukuran evaluasi performa prediksi hotspot.
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            # `tf.keras.metrics.Precision(name` menyimpan ukuran evaluasi performa prediksi hotspot.
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            # `tf.keras.metrics.Recall(name` menyimpan ukuran evaluasi performa prediksi hotspot.
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
            # `tf.keras.metrics.AUC(name` menyimpan ukuran evaluasi performa prediksi hotspot.
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    # Hasil ini dikembalikan sebagai output fungsi `build_model` untuk tahap berikutnya.
    return model


# Fungsi `capture_model_summary` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def capture_model_summary(model) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model.summary()
    # Hasil ini dikembalikan sebagai output fungsi `capture_model_summary` untuk tahap berikutnya.
    return buffer.getvalue().strip()


# Menggabungkan hasil proses menjadi ringkasan JSON/console untuk laporan dan verifikasi eksperimen.
def build_summary(args: argparse.Namespace) -> dict:
    channels = feature_stack_channels(args.feature_stack)
    # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
    summary: dict = {
        "phase": "modeling",
        "model_profile": "historical_risk_patch_160",
        "seq_length": args.seq_length,
        "patch_width": args.patch_width,
        "patch_height": args.patch_height,
        "feature_stack": args.feature_stack,
        "channels": channels,
        "loss_strategy": args.loss_strategy,
        "context_kernel": args.context_kernel,
        "positive_class_weight": args.pos_weight,
        "optimizer": {"name": "Adam", "learning_rate": 0.001},
        "metrics": ["accuracy", "precision", "recall", "pr_auc"],
        # Memulai perintah yang dilanjutkan pada baris berikutnya agar lebih rapi.
        "static_layers": build_static_layer_specs(
            channels=channels,
            seq_length=args.seq_length,
            # `patch_height` berkaitan dengan potongan citra 160x160 yang dipakai model.
            patch_height=args.patch_height,
            # `patch_width` berkaitan dengan potongan citra 160x160 yang dipakai model.
            patch_width=args.patch_width,
        ),
        "tensorflow_available": False,
        "tensorflow_error": None,
        "model_instantiated": False,
        "parameter_count": None,
        "model_summary_text": None,
    }

    if args.skip_tensorflow:
        summary["tensorflow_error"] = "skip_tensorflow=True"
        # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
        return summary

    tf, tf_error = import_tensorflow()
    if tf is None:
        summary["tensorflow_error"] = tf_error
        # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
        return summary

    summary["tensorflow_available"] = True
    model = build_model(
        seq_length=args.seq_length,
        # `patch_height` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_height=args.patch_height,
        # `patch_width` berkaitan dengan potongan citra 160x160 yang dipakai model.
        patch_width=args.patch_width,
        channels=channels,
        # Bobot kelas positif agar loss lebih memperhatikan piksel hotspot yang langka.
        pos_weight=args.pos_weight,
        loss_strategy=args.loss_strategy,
        context_kernel=args.context_kernel,
        tf=tf,
    )
    summary["model_instantiated"] = True
    summary["parameter_count"] = int(model.count_params())
    summary["model_summary_text"] = capture_model_summary(model)

    if args.save_summary:
        with open(args.save_summary, "w", encoding="utf-8") as handle:
            handle.write(str(summary["model_summary_text"]))
            handle.write("\n")
        summary["model_summary_saved_to"] = args.save_summary

    # Hasil ini dikembalikan sebagai output fungsi `build_summary` untuk tahap berikutnya.
    return summary


# Fungsi `print_human_summary` menjalankan bagian khusus dari pipeline hotspot sesuai nama prosesnya.
def print_human_summary(summary: dict) -> None:
    print("[modeling] Profile model:", summary["model_profile"])
    print("[modeling] Seq length:", summary["seq_length"])
    print("[modeling] Patch size:", f"{summary['patch_width']}x{summary['patch_height']}")
    print("[modeling] Feature stack:", summary["feature_stack"])
    print("[modeling] Channels:", summary["channels"])
    print("[modeling] Loss strategy:", summary["loss_strategy"])
    print("[modeling] Context kernel:", summary["context_kernel"])
    print("[modeling] Positive class weight:", f"{summary['positive_class_weight']:.4f}")
    print("[modeling] Optimizer:", f"{summary['optimizer']['name']} lr={summary['optimizer']['learning_rate']}")
    print("[modeling] Metrics:", ", ".join(summary["metrics"]))
    print("[modeling] Layer utama:")
    for layer in summary["static_layers"]:
        print(f"- {layer['name']} | {layer['type']}")

    if summary["tensorflow_available"] and summary["model_instantiated"]:
        print("[modeling] TensorFlow: tersedia")
        print("[modeling] Parameter model:", summary["parameter_count"])
        print()
        print(summary["model_summary_text"])
    # Menjalankan pilihan cadangan ketika syarat sebelumnya tidak terpenuhi.
    else:
        print("[modeling] TensorFlow: tidak tersedia atau dilewati")
        print("[modeling] Detail:", summary["tensorflow_error"])


# Menjadi titik masuk script saat dijalankan dari terminal.
def main() -> None:
    # Parser command line menampung parameter eksperimen agar proses bisa direplikasi.
    parser = build_parser()
    # Argumen runtime yang menentukan dataset, split, patch, threshold, dan output.
    args = parser.parse_args()
    # Ringkasan akhir proses yang dicetak atau disimpan untuk kebutuhan BAB IV.
    summary = build_summary(args)
    print_human_summary(summary)
    if args.json:
        print()
        # Hasil ringkasan atau metadata disimpan agar bisa dipakai di laporan/validasi ulang.
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
