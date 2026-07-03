
// Komentar file skripsi: Frontend dashboard: mengatur upload 7 citra H-6 sampai H0, memanggil API prediksi, menampilkan hasil, dan riwayat.
// Konteks laporan: mendukung proses eksperimen/training pada BAB IV.

// Mengambil konfigurasi dari template Jinja, seperti daftar H-6..H0 dan threshold rekomendasi.
const appConfig = window.APP_CONFIG || {};
// Daftar nama file wajib dari backend: H-6, H-5, H-4, H-3, H-2, H-1, dan H0.
const requiredStems = appConfig.requiredStems || [];

// Form utama yang mengirim tujuh citra historis ke endpoint prediksi.
const form = document.getElementById("predict-form");
// Input file tempat pengguna memilih 7 citra hotspot historis.
const filesInput = document.getElementById("files");
// Kolom threshold yang menentukan batas binary mask area risiko.
const thresholdInput = document.getElementById("threshold");
// Kolom horizon yang ditahan pada H+1 sesuai batasan sistem.
const horizonInput = document.getElementById("horizon");
// Tombol untuk memulai upload dan inferensi model.
const predictButton = document.getElementById("predict-button");
// Tombol untuk mengosongkan form dan hasil prediksi.
const resetButton = document.getElementById("reset-button");
// Elemen ringkasan jumlah file yang dipilih pengguna.
const fileCounter = document.getElementById("file-counter");
// Daftar visual file H-6 sampai H0 yang akan dikirim.
const fileList = document.getElementById("file-list");
// Kotak pesan validasi jika file, tanggal, atau response API tidak valid.
const errorBox = document.getElementById("error-box");

// Bagian halaman yang menampilkan output setelah prediksi berhasil.
const resultSection = document.getElementById("result-section");
// Preview citra input acuan, biasanya H0.
const resultInput = document.getElementById("result-input");
// Gambar heatmap probability map H+1 dari backend.
const resultHeatmap = document.getElementById("result-heatmap");
// Gambar overlay prediksi dengan batas/nomor wilayah administrasi.
const resultOverlay = document.getElementById("result-overlay");
// Gambar overlay bersih tanpa batas administrasi untuk membaca risiko.
const resultOverlayPlain = document.getElementById("result-overlay-plain");
// Panel pembungkus overlay bersih.
const resultOverlayPlainPanel = document.getElementById("result-overlay-plain-panel");
// Gambar binary mask berdasarkan threshold risiko.
const resultBinary = document.getElementById("result-binary");
// Panel pembungkus binary mask.
const resultBinaryPanel = document.getElementById("result-binary-panel");
// Catatan yang menjelaskan arti putih/hitam pada binary mask.
const resultBinaryNote = document.getElementById("result-binary-note");
// Container legenda nomor/warna kabupaten/kota pada overlay.
const resultMapLegend = document.getElementById("result-map-legend");
// Toggle untuk menampilkan atau menyembunyikan nomor wilayah pada overlay.
const regionNumberToggle = document.getElementById("region-number-toggle");
// Area metadata hasil seperti model, threshold, durasi, dan target date.
const resultMeta = document.getElementById("result-meta");
// Daftar link output H+1 seperti heatmap, overlay, binary, dan file lain.
const outputList = document.getElementById("output-list");
// Link unduh heatmap hasil prediksi.
const downloadHeatmap = document.getElementById("download-heatmap");
// Link unduh overlay wilayah sesuai pilihan bernomor/tanpa nomor.
const downloadOverlay = document.getElementById("download-overlay");
// Link unduh overlay bersih.
const downloadOverlayPlain = document.getElementById("download-overlay-plain");
// Link unduh binary mask.
const downloadBinary = document.getElementById("download-binary");
// Link unduh ZIP berisi input, output, probability, dan metadata.
const downloadZip = document.getElementById("download-zip");

// Body tabel riwayat prediksi yang diisi dari /api/predictions.
const historyBody = document.getElementById("history-body");
// Ringkasan jumlah/durasi riwayat prediksi.
const historySummary = document.getElementById("history-summary");
// Tombol untuk memuat ulang riwayat prediksi.
const refreshHistoryButton = document.getElementById("refresh-history");

// Tombol untuk memuat ulang status runtime model.
const refreshRuntimeButton = document.getElementById("refresh-runtime");
// Badge status yang menunjukkan ConvLSTM aktif atau fallback heuristic.
const runtimeStateBadge = document.getElementById("runtime-state-badge");
// Teks penjelas status runtime model.
const runtimeStateText = document.getElementById("runtime-state-text");
// Elemen untuk menampilkan profile model aktif.
const runtimeModelProfile = document.getElementById("runtime-model-profile");
// Elemen untuk menampilkan nama model aktif.
const runtimeModelName = document.getElementById("runtime-model-name");
// Elemen untuk menampilkan backend inferensi yang benar-benar dipakai.
const runtimeEngine = document.getElementById("runtime-engine");
// Elemen untuk menampilkan mode preprocessing input.
const runtimePreprocess = document.getElementById("runtime-preprocess");
// Elemen untuk menampilkan ukuran input/model.
const runtimeGrid = document.getElementById("runtime-grid");
// Elemen untuk menampilkan ukuran patch dan stride.
const runtimePatch = document.getElementById("runtime-patch");
// Elemen untuk menampilkan threshold rekomendasi model.
const runtimeThreshold = document.getElementById("runtime-threshold");
// Elemen untuk menampilkan versi TensorFlow runtime.
const runtimeTfVersion = document.getElementById("runtime-tf-version");
// Elemen untuk menampilkan kandidat file model yang dicoba backend.
const runtimeCandidates = document.getElementById("runtime-candidates");

// Kartu statistik jumlah prediksi tersimpan.
const statTotalPredictions = document.getElementById("stat-total-predictions");
const statTotalNote = document.getElementById("stat-total-note");
// Kartu statistik waktu prediksi terbaru.
const statLatestRun = document.getElementById("stat-latest-run");
const statLatestNote = document.getElementById("stat-latest-note");
// Kartu statistik rata-rata durasi inferensi.
const statAvgProcessing = document.getElementById("stat-avg-processing");
const statAvgNote = document.getElementById("stat-avg-note");
// Kartu statistik engine aktif: ConvLSTM atau fallback.
const statActiveEngine = document.getElementById("stat-active-engine");
const statEngineNote = document.getElementById("stat-engine-note");

// Placeholder saat belum ada riwayat prediksi.
const spotlightEmpty = document.getElementById("spotlight-empty");
// Panel sorotan hasil prediksi terbaru.
const spotlightContent = document.getElementById("spotlight-content");
// Preview heatmap prediksi terbaru.
const spotlightHeatmap = document.getElementById("spotlight-heatmap");
// Preview overlay prediksi terbaru.
const spotlightOverlay = document.getElementById("spotlight-overlay");
// Judul sorotan prediksi terbaru.
const spotlightTitle = document.getElementById("spotlight-title");
// Metadata singkat sorotan prediksi terbaru.
const spotlightMeta = document.getElementById("spotlight-meta");
// Tombol untuk membuka detail prediksi yang sedang disorot.
const spotlightOpenButton = document.getElementById("spotlight-open");

let currentRuntimeStatus = null;
let spotlightPredictionId = null;
// `currentOverlayWithNumbersUrl` menyimpan URL overlay aktif agar toggle nomor wilayah tidak perlu memanggil API ulang.
let currentOverlayWithNumbersUrl = "";
// `currentOverlayWithoutNumbersUrl` menyimpan URL overlay aktif agar toggle nomor wilayah tidak perlu memanggil API ulang.
let currentOverlayWithoutNumbersUrl = "";
// `currentOverlayWithNumbersDownload` menyimpan URL overlay aktif agar toggle nomor wilayah tidak perlu memanggil API ulang.
let currentOverlayWithNumbersDownload = "";
// `currentOverlayWithoutNumbersDownload` menyimpan URL overlay aktif agar toggle nomor wilayah tidak perlu memanggil API ulang.
let currentOverlayWithoutNumbersDownload = "";

// Menyeragamkan label file agar variasi seperti H-0 dibaca sebagai H0.
function normalizeStemPart(value) {
  const clean = String(value || "").trim().toUpperCase().replace(/\s+/g, "");
  if (clean === "H-0") return "H0";
  // Nilai ini menjadi hasil fungsi `normalizeStemPart` untuk proses tampilan berikutnya.
  return clean;
}

// Mengambil nama file tanpa folder dan ekstensi sebelum dicek sebagai H-6..H0.
function fileBaseName(name) {
  // `fileName` menyimpan file upload atau nama file citra historis.
  const fileName = String(name || "").split(/[\\/]/).pop() || "";
  const dotIndex = fileName.lastIndexOf(".");
  const withoutExtension = dotIndex >= 0 ? fileName.slice(0, dotIndex) : fileName;
  // Nilai ini menjadi hasil fungsi `fileBaseName` untuk proses tampilan berikutnya.
  return withoutExtension.trim().replace(/\s+/g, "");
}

// Memastikan tanggal pada nama file memakai format YYYY-MM-DD yang valid.
function isValidIsoDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  // Nilai ini menjadi hasil fungsi `isValidIsoDate` untuk proses tampilan berikutnya.
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

// Menghitung tanggal target H+1 dari tanggal H0.
function addDaysIso(value, days) {
  const parsed = new Date(`${value}T00:00:00Z`);
  parsed.setUTCDate(parsed.getUTCDate() + days);
  // Nilai ini menjadi hasil fungsi `addDaysIso` untuk proses tampilan berikutnya.
  return parsed.toISOString().slice(0, 10);
}

// Membaca identitas input dari nama file, termasuk stem H-n dan tanggal opsional.
function parseInputIdentity(name) {
  const base = fileBaseName(name);
  const parts = base.split("_");
  if (parts.length > 2) {
    // Nilai ini menjadi hasil fungsi `parseInputIdentity` untuk proses tampilan berikutnya.
    return {
      stem: "",
      date: null,
      valid: false,
      error: `Nama file '${name}' tidak valid. Gunakan H-6.png atau H-6_YYYY-MM-DD.png.`,
    };
  }

  const stem = normalizeStemPart(parts[0]);
  if (parts.length === 1) {
    // Nilai ini menjadi hasil fungsi `parseInputIdentity` untuk proses tampilan berikutnya.
    return { stem, date: null, valid: true, error: "" };
  }

  // `dateText` menyimpan tanggal input atau target H+1.
  const dateText = parts[1];
  if (!isValidIsoDate(dateText)) {
    // Nilai ini menjadi hasil fungsi `parseInputIdentity` untuk proses tampilan berikutnya.
    return {
      stem,
      date: null,
      valid: false,
      error: `Tanggal pada file '${name}' harus memakai format YYYY-MM-DD.`,
    };
  }

  // Nilai ini menjadi hasil fungsi `parseInputIdentity` untuk proses tampilan berikutnya.
  return { stem, date: dateText, valid: true, error: "" };
}

// Mengambil stem H-6..H0 dari nama file untuk validasi urutan upload.
function normalizeStem(name) {
  // Nilai ini menjadi hasil fungsi `normalizeStem` untuk proses tampilan berikutnya.
  return parseInputIdentity(name).stem;
}

// Mengubah FileList browser menjadi array agar mudah divalidasi dan ditampilkan.
function getSelectedFiles() {
  // Nilai ini menjadi hasil fungsi `getSelectedFiles` untuk proses tampilan berikutnya.
  return Array.from(filesInput.files || []);
}

// Membersihkan teks dari backend/user sebelum dimasukkan ke HTML agar aman.
function escapeHtml(value) {
  // Nilai ini menjadi hasil fungsi `escapeHtml` untuk proses tampilan berikutnya.
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Mengisi elemen teks dengan fallback '-' jika metadata prediksi kosong.
function setText(node, value, fallback = "-") {
  if (!node) return;
  const resolved = value === null || value === undefined || value === "" ? fallback : value;
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  node.textContent = String(resolved);
}

// Mengatur src gambar hasil, atau menghapus src saat tidak ada output.
function setImageSource(node, url) {
  if (!node) return;
  if (url) {
    // Sumber gambar diarahkan ke heatmap/overlay/binary hasil prediksi.
    node.src = url;
  } else {
    node.removeAttribute("src");
  }
}

// Mengganti overlay antara versi bernomor wilayah dan tanpa nomor tanpa memanggil API ulang.
function updateRegionNumberOverlay() {
  // `hasNumberlessOverlay` menyimpan URL overlay hasil prediksi.
  const hasNumberlessOverlay = Boolean(currentOverlayWithoutNumbersUrl);
  const showNumbers = !regionNumberToggle || regionNumberToggle.checked || !hasNumberlessOverlay;
  const imageUrl = showNumbers
    ? currentOverlayWithNumbersUrl
    : currentOverlayWithoutNumbersUrl;
  const fallbackUrl = currentOverlayWithNumbersUrl || currentOverlayWithoutNumbersUrl;
  setImageSource(resultOverlay, imageUrl || fallbackUrl);

  const downloadUrl = showNumbers
    ? currentOverlayWithNumbersDownload
    : currentOverlayWithoutNumbersDownload;
  const fallbackDownload = currentOverlayWithNumbersDownload || currentOverlayWithoutNumbersDownload;
  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadOverlay.href = downloadUrl || fallbackDownload || "#";
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  downloadOverlay.textContent = showNumbers ? "Unduh Overlay Bernomor" : "Unduh Overlay Tanpa Nomor";
  // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
  downloadOverlay.classList.toggle("disabled", !(downloadUrl || fallbackDownload));

  if (regionNumberToggle) {
    regionNumberToggle.disabled = !hasNumberlessOverlay;
    const toggleWrap = regionNumberToggle.closest(".map-toggle");
    // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
    if (toggleWrap) toggleWrap.classList.toggle("is-disabled", !hasNumberlessOverlay);
  }
}

// Menyembunyikan pesan error setelah input atau response sudah valid.
function hideError() {
  errorBox.classList.add("d-none");
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  errorBox.textContent = "";
}

// Menampilkan pesan error validasi file atau kegagalan API.
function showError(message) {
  errorBox.classList.remove("d-none");
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  errorBox.textContent = message;
}

// Mengubah timestamp backend menjadi format tanggal Indonesia.
function formatDate(value) {
  if (!value) return "-";
  // `date` menyimpan tanggal input atau target H+1.
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  // Nilai ini menjadi hasil fungsi `formatDate` untuk proses tampilan berikutnya.
  return date.toLocaleString("id-ID", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

// Mengubah durasi inferensi milidetik menjadi teks ms/detik.
function formatDuration(ms) {
  const numeric = Number(ms);
  if (!Number.isFinite(numeric)) return "-";
  if (numeric >= 1000) {
    // Nilai ini menjadi hasil fungsi `formatDuration` untuk proses tampilan berikutnya.
    return `${(numeric / 1000).toFixed(1)} dtk`;
  }
  // Nilai ini menjadi hasil fungsi `formatDuration` untuk proses tampilan berikutnya.
  return `${Math.round(numeric)} ms`;
}

// Menampilkan threshold dengan dua angka desimal.
function formatThreshold(value) {
  if (value === null || value === undefined || value === "") return "-";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  // Nilai ini menjadi hasil fungsi `formatThreshold` untuk proses tampilan berikutnya.
  return numeric.toFixed(2);
}

// Meringkas ukuran patch dan stride dari payload runtime/prediksi.
function summarizePatchInfo(payload) {
  const patchSize = payload?.patch_size;
  const patchStride = payload?.patch_stride;
  if (!patchSize && !patchStride) return "Tidak digunakan";
  const sizeText = patchSize ? `${patchSize} px` : "-";
  const strideText = patchStride ? `${patchStride} px` : "-";
  // Nilai ini menjadi hasil fungsi `summarizePatchInfo` untuk proses tampilan berikutnya.
  return `${sizeText} / ${strideText}`;
}

// Meringkas ukuran grid/output probability map.
function summarizeGridInfo(payload) {
  const gridSize = payload?.grid_size ?? appConfig.gridSize;
  const channels = payload?.channels ?? "-";
  const timeSteps = payload?.time_steps ?? requiredStems.length;
  if (!gridSize) return "-";
  // Nilai ini menjadi hasil fungsi `summarizeGridInfo` untuk proses tampilan berikutnya.
  return `${gridSize} x ${gridSize} | ${channels} ch | ${timeSteps} frame`;
}

// Meringkas kandidat model yang ditemukan backend.
function summarizeCandidates(candidates) {
  if (!Array.isArray(candidates) || candidates.length === 0) return "Tidak ada kandidat model yang terdeteksi.";
  // Nilai ini menjadi hasil fungsi `summarizeCandidates` untuk proses tampilan berikutnya.
  return candidates
    .slice(0, 4)
    .map((item) => String(item).split(/[\\/]/).pop())
    .join(" | ");
}

// Meringkas nama backend agar card statistik tidak penuh oleh nama file model yang panjang.
function formatEngineLabel(engineName) {
  const value = String(engineName || "").toLowerCase();
  if (!value || value === "-") return "-";
  if (value.includes("heuristic")) return "Fallback Heuristik";
  if (value.includes("convlstm") || value.includes("tensorflow")) return "ConvLSTM TensorFlow";
  return String(engineName);
}

// Mengambil pesan error terbaik dari response API.
function responseErrorText(payload) {
  if (!payload) return "Terjadi kesalahan pada server.";
  if (typeof payload.detail === "string") return payload.detail;
  if (Array.isArray(payload.detail)) {
    // Nilai ini menjadi hasil fungsi `responseErrorText` untuk proses tampilan berikutnya.
    return payload.detail.map((item) => item.msg || JSON.stringify(item)).join(", ");
  }
  // Nilai ini menjadi hasil fungsi `responseErrorText` untuk proses tampilan berikutnya.
  return JSON.stringify(payload);
}

// Mengosongkan panel hasil agar dashboard kembali ke kondisi awal.
function resetResultSection() {
  resultSection.classList.add("d-none");
  currentOverlayWithNumbersUrl = "";
  currentOverlayWithoutNumbersUrl = "";
  currentOverlayWithNumbersDownload = "";
  currentOverlayWithoutNumbersDownload = "";
  if (regionNumberToggle) {
    regionNumberToggle.checked = true;
    regionNumberToggle.disabled = true;
    const toggleWrap = regionNumberToggle.closest(".map-toggle");
    if (toggleWrap) toggleWrap.classList.add("is-disabled");
  }
  setImageSource(resultInput, "");
  setImageSource(resultHeatmap, "");
  setImageSource(resultOverlay, "");
  setImageSource(resultOverlayPlain, "");
  setImageSource(resultBinary, "");
  if (resultOverlayPlainPanel) resultOverlayPlainPanel.classList.add("d-none");
  if (resultBinaryPanel) resultBinaryPanel.classList.add("d-none");
  if (resultBinaryNote) {
    // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
    resultBinaryNote.textContent =
      "Area putih berarti probabilitas model melewati threshold. Area hitam berarti di bawah threshold.";
  }
  // HTML panel/tabel dirender ulang dari payload backend.
  if (resultMapLegend) resultMapLegend.innerHTML = "";
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  resultMeta.textContent = "";
  // HTML panel/tabel dirender ulang dari payload backend.
  outputList.innerHTML = "";

  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadHeatmap.href = "#";
  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadOverlay.href = "#";
  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadOverlayPlain.href = "#";
  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadBinary.href = "#";
  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadZip.href = "#";

  downloadHeatmap.classList.add("disabled");
  downloadOverlay.classList.add("disabled");
  downloadOverlayPlain.classList.add("disabled");
  downloadBinary.classList.add("disabled");
  downloadZip.classList.add("disabled");
}

// Menghapus pilihan file, error, dan hasil prediksi dari form.
function resetFormState() {
  form.reset();
  // HTML panel/tabel dirender ulang dari payload backend.
  fileList.innerHTML = "";
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  fileCounter.textContent = "0/7 file terunggah";
  hideError();
  resetResultSection();
  validateSelection();
}

// Memvalidasi 7 file H-6 sampai H0 dan tanggal harian sebelum dikirim ke backend.
function validateSelection() {
  hideError();
  // `files` menyimpan file upload atau nama file citra historis.
  const files = getSelectedFiles();
  const identities = files.map((file) => ({ file, ...parseInputIdentity(file.name) }));
  const invalidIdentity = identities.find((item) => !item.valid);
  const stems = identities.map((item) => item.stem);
  const uniqueStems = new Set(stems);

  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  fileCounter.textContent = `${files.length}/7 file terunggah`;
  // HTML panel/tabel dirender ulang dari payload backend.
  fileList.innerHTML = "";
  for (const file of files) {
    const item = document.createElement("li");
    // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
    item.textContent = file.name;
    fileList.appendChild(item);
  }

  // Cabang ini menangani validasi jumlah/urutan file upload citra historis.
  if (files.length !== 7) {
    predictButton.disabled = true;
    return;
  }

  if (invalidIdentity) {
    showError(invalidIdentity.error);
    predictButton.disabled = true;
    return;
  }

  if (uniqueStems.size !== 7) {
    showError("Terdapat nama file yang duplikat.");
    predictButton.disabled = true;
    return;
  }

  for (const expected of requiredStems) {
    if (!uniqueStems.has(expected)) {
      showError(`Nama file tidak lengkap. Pastikan ada file ${expected}.`);
      predictButton.disabled = true;
      return;
    }
  }

  // `datedCount` menyimpan tanggal input atau target H+1.
  const datedCount = identities.filter((item) => item.date).length;
  if (datedCount > 0) {
    if (datedCount !== identities.length) {
      showError("Jika memakai tanggal, semua file H-6 sampai H0 harus mencantumkan tanggal.");
      predictButton.disabled = true;
      return;
    }

    const identityByStem = new Map(identities.map((item) => [item.stem, item]));
    // `startDate` menyimpan tanggal input atau target H+1.
    const startDate = identityByStem.get(requiredStems[0])?.date;
    if (!startDate) {
      showError("Tanggal H-6 wajib tersedia jika format bertanggal digunakan.");
      predictButton.disabled = true;
      return;
    }

    for (const [offset, stem] of requiredStems.entries()) {
      // `expectedDate` menyimpan tanggal input atau target H+1.
      const expectedDate = addDaysIso(startDate, offset);
      // `actualDate` menyimpan tanggal input atau target H+1.
      const actualDate = identityByStem.get(stem)?.date;
      if (actualDate !== expectedDate) {
        showError(`Tanggal file ${stem} harus ${expectedDate}, tetapi ditemukan ${actualDate || "-"}.`);
        predictButton.disabled = true;
        return;
      }
    }
  }

  predictButton.disabled = false;
}

// Menampilkan status model aktif, threshold, TensorFlow, dan fallback runtime.
function renderRuntimeStatus(payload) {
  currentRuntimeStatus = payload;

  const engineName = payload?.prediction_engine || payload?.model_backend || "-";
  const usingFallback = String(engineName).toLowerCase().includes("heuristic");

  runtimeStateBadge.className = `status-badge ${usingFallback ? "status-warning" : "status-success"}`;
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  runtimeStateBadge.textContent = usingFallback ? "Fallback Heuristik" : "ConvLSTM Aktif";
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  runtimeStateText.textContent = usingFallback
    ? "Aplikasi tetap berjalan, tetapi inferensi saat ini menggunakan backend fallback."
    : "Model ConvLSTM berhasil dimuat dan siap dipakai untuk prediksi.";

  setText(runtimeModelProfile, payload?.model_profile || appConfig.modelProfile);
  setText(runtimeModelName, payload?.model_display_name || appConfig.modelDisplayName);
  setText(runtimeEngine, engineName);
  setText(runtimePreprocess, payload?.preprocess_mode);
  setText(runtimeGrid, summarizeGridInfo(payload));
  setText(runtimePatch, summarizePatchInfo(payload));
  setText(runtimeThreshold, formatThreshold(payload?.recommended_threshold));
  setText(runtimeTfVersion, payload?.tensorflow_version || "Tidak tersedia");
  setText(runtimeCandidates, summarizeCandidates(payload?.model_candidates));

  setText(statActiveEngine, formatEngineLabel(engineName));
  setText(
    statEngineNote,
    `${payload?.model_profile || appConfig.modelProfile || "-"} | ${payload?.inference_mode || appConfig.inferenceMode || "-"}`,
  );
}

// Menampilkan kondisi gagal mengambil status runtime.
function renderRuntimeError(message) {
  runtimeStateBadge.className = "status-badge status-danger";
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  runtimeStateBadge.textContent = "Gagal Memuat";
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  runtimeStateText.textContent = message;

  setText(runtimeModelProfile, "-");
  setText(runtimeModelName, "-");
  setText(runtimeEngine, "-");
  setText(runtimePreprocess, "-");
  setText(runtimeGrid, "-");
  setText(runtimePatch, "-");
  setText(runtimeThreshold, "-");
  setText(runtimeTfVersion, "-");
  setText(runtimeCandidates, "-");

  setText(statActiveEngine, "-");
  setText(statEngineNote, "Status runtime belum tersedia");
}

// Memanggil /api/runtime/status untuk mengetahui engine model yang sedang dipakai.
async function loadRuntimeStatus() {
  try {
    // Response ini berasal dari endpoint backend prediksi/status/riwayat.
    const response = await fetch("/api/runtime/status");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(responseErrorText(payload));
    }
    renderRuntimeStatus(payload);
  } catch (error) {
    renderRuntimeError(error.message || "Gagal mengambil status runtime.");
  }
}

// Menampilkan keadaan kosong saat belum ada riwayat prediksi.
function renderSpotlightEmpty() {
  spotlightPredictionId = null;
  spotlightEmpty.classList.remove("d-none");
  spotlightContent.classList.add("d-none");
  spotlightOpenButton.classList.add("d-none");
  setImageSource(spotlightHeatmap, "");
  setImageSource(spotlightOverlay, "");
  setText(spotlightTitle, "-");
  setText(spotlightMeta, "Belum ada hasil yang dapat disorot.");
}

// Menyusun metadata singkat untuk prediksi terbaru.
function spotlightMetaText(payload) {
  // `targetDateText` menyimpan tanggal input atau target H+1.
  const targetDateText = payload.target_date ? `Tanggal prediksi: ${payload.target_date}` : null;
  const parts = [
    `Waktu: ${formatDate(payload.created_at)}`,
    `Model: ${payload.model_profile || "-"}`,
    `Backend: ${payload.prediction_engine || payload.model_backend || "-"}`,
    `Prediksi: H+${payload.horizon ?? "-"}`,
    targetDateText,
    `Threshold: ${formatThreshold(payload.threshold)}`,
    `Proses: ${formatDuration(payload.processing_time_ms)}`,
  ].filter(Boolean);
  // Nilai ini menjadi hasil fungsi `spotlightMetaText` untuk proses tampilan berikutnya.
  return parts.join(" | ");
}

// Menampilkan heatmap dan overlay prediksi terbaru pada panel sorotan.
function renderSpotlight(payload) {
  const firstOutput = (payload?.outputs || [])[0] || {};
  const stamp = `t=${Date.now()}`;
  // `heatmapUrl` menyimpan URL/elemen heatmap probability map.
  const heatmapUrl = firstOutput.heatmap_url || payload?.heatmap_url || "";
  // `overlayUrl` menyimpan URL overlay hasil prediksi.
  const overlayUrl = firstOutput.overlay_url || payload?.overlay_url || "";
  if (!heatmapUrl && !overlayUrl) {
    renderSpotlightEmpty();
    return;
  }

  spotlightPredictionId = payload?.prediction_id || null;
  spotlightEmpty.classList.add("d-none");
  spotlightContent.classList.remove("d-none");
  // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
  spotlightOpenButton.classList.toggle("d-none", !spotlightPredictionId);

  setImageSource(spotlightHeatmap, heatmapUrl ? `${heatmapUrl}?${stamp}` : "");
  setImageSource(spotlightOverlay, overlayUrl ? `${overlayUrl}?${stamp}` : "");
  setText(spotlightTitle, `Prediksi ${payload?.prediction_id || "-"}`);
  setText(spotlightMeta, spotlightMetaText(payload));
}

// Menghitung statistik ringkas dari daftar riwayat prediksi.
function renderDashboardStats(items) {
  if (!Array.isArray(items) || items.length === 0) {
    setText(statTotalPredictions, "0");
    setText(statTotalNote, "Belum ada riwayat tersimpan");
    setText(statLatestRun, "-");
    setText(statLatestNote, "Menunggu prediksi pertama");
    setText(statAvgProcessing, "-");
    setText(statAvgNote, "Belum ada data waktu proses");
    if (!currentRuntimeStatus) {
      setText(statActiveEngine, "-");
      setText(statEngineNote, "Status runtime belum tersedia");
    }
    setText(historySummary, "Belum ada riwayat prediksi yang tersimpan.");
    renderSpotlightEmpty();
    return;
  }

  const latest = items[0];
  const processingValues = items
    .map((item) => Number(item.processing_time_ms))
    .filter((value) => Number.isFinite(value));
  const averageMs = processingValues.length
    ? Math.round(processingValues.reduce((sum, value) => sum + value, 0) / processingValues.length)
    : null;

  setText(statTotalPredictions, String(items.length));
  setText(statTotalNote, "Riwayat terbaru yang berhasil dimuat");
  setText(statLatestRun, formatDate(latest.created_at));
  setText(statLatestNote, `${latest.model_profile || "-"} | H+${latest.horizon ?? "-"}`);
  setText(statAvgProcessing, formatDuration(averageMs));
  setText(statAvgNote, `${processingValues.length || 0} data waktu proses tersedia`);

  if (!currentRuntimeStatus) {
    setText(statActiveEngine, formatEngineLabel(latest.prediction_engine || latest.model_backend || "-"));
    setText(statEngineNote, "Mengikuti riwayat prediksi terbaru");
  }

  setText(
    historySummary,
    `Menampilkan ${items.length} riwayat terbaru. Prediksi terakhir dibuat pada ${formatDate(latest.created_at)}.`,
  );
  renderSpotlight(latest);
}

// Menampilkan legenda kabupaten/kota yang dikirim backend.
function renderMapLegend(regions) {
  if (!resultMapLegend) return;
  const items = Array.isArray(regions) ? regions : [];
  if (!items.length) {
    // HTML panel/tabel dirender ulang dari payload backend.
    resultMapLegend.innerHTML = `<span class="map-legend-empty">Legenda wilayah belum tersedia.</span>`;
    return;
  }

  // HTML panel/tabel dirender ulang dari payload backend.
  resultMapLegend.innerHTML = "";
  for (const item of items) {
    const chip = document.createElement("div");
    chip.className = "map-legend-item";
    const color = item.color || "#999999";
    const name = item.name || "Wilayah";
    const number = item.number || "-";
    // HTML panel/tabel dirender ulang dari payload backend.
    chip.innerHTML = `
      <span class="map-legend-number" style="background:${escapeHtml(color)}">${escapeHtml(number)}</span>
      <span>${escapeHtml(name)}</span>
    `;
    resultMapLegend.appendChild(chip);
  }
}

// Menampilkan semua output prediksi: input, overlay, heatmap, binary, metadata, dan link unduh.
function renderResult(payload, files = null) {
  const stamp = `t=${Date.now()}`;
  // `selectedFiles` menyimpan file upload atau nama file citra historis.
  const selectedFiles = Array.isArray(files) ? files : [];
  // `h0File` menyimpan file upload atau nama file citra historis.
  const h0File = selectedFiles.find((f) => normalizeStem(f.name) === "H0");
  const firstOutput = (payload.outputs || [])[0] || {};
  // `heatmapUrl` menyimpan URL/elemen heatmap probability map.
  const heatmapUrl = payload.heatmap_url || firstOutput.heatmap_url || "";
  // `overlayUrl` menyimpan URL overlay hasil prediksi.
  const overlayUrl = payload.overlay_url || firstOutput.overlay_url || "";
  // `overlayNoNumbersUrl` menyimpan URL overlay hasil prediksi.
  const overlayNoNumbersUrl = payload.overlay_no_numbers_url || firstOutput.overlay_no_numbers_url || "";
  // `overlayPlainUrl` menyimpan URL overlay hasil prediksi.
  const overlayPlainUrl = payload.overlay_plain_url || firstOutput.overlay_plain_url || "";
  const binaryUrl = payload.binary_url || firstOutput.binary_url || "";
  const zipUrl = payload.download_zip_url || `/api/predictions/${payload.prediction_id}/download`;
  // `modelProfile` menyimpan file upload atau nama file citra historis.
  const modelProfile = payload.model_profile || appConfig.modelProfile || "-";
  // `recommendedThreshold` menyimpan nilai threshold yang dipakai membaca binary mask.
  const recommendedThreshold = payload.recommended_threshold ?? appConfig.recommendedThreshold ?? "-";
  // `usedThreshold` menyimpan nilai threshold yang dipakai membaca binary mask.
  const usedThreshold = payload.threshold ?? thresholdInput?.value ?? "";

  resultSection.classList.remove("d-none");
  if (h0File) {
    // Sumber gambar diarahkan ke heatmap/overlay/binary hasil prediksi.
    resultInput.src = URL.createObjectURL(h0File);
  } else if (payload.input_urls?.H0) {
    // Sumber gambar diarahkan ke heatmap/overlay/binary hasil prediksi.
    resultInput.src = `${payload.input_urls.H0}?${stamp}`;
  } else {
    resultInput.removeAttribute("src");
  }
  setImageSource(resultHeatmap, heatmapUrl ? `${heatmapUrl}?${stamp}` : "");
  currentOverlayWithNumbersUrl = overlayUrl ? `${overlayUrl}?${stamp}` : "";
  currentOverlayWithoutNumbersUrl = overlayNoNumbersUrl ? `${overlayNoNumbersUrl}?${stamp}` : "";
  currentOverlayWithNumbersDownload = overlayUrl || "";
  currentOverlayWithoutNumbersDownload = overlayNoNumbersUrl || "";
  updateRegionNumberOverlay();
  setImageSource(resultOverlayPlain, overlayPlainUrl ? `${overlayPlainUrl}?${stamp}` : "");
  // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
  if (resultOverlayPlainPanel) resultOverlayPlainPanel.classList.toggle("d-none", !overlayPlainUrl);
  setImageSource(resultBinary, binaryUrl ? `${binaryUrl}?${stamp}` : "");
  // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
  if (resultBinaryPanel) resultBinaryPanel.classList.toggle("d-none", !binaryUrl);
  if (resultBinaryNote) {
    // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
    resultBinaryNote.textContent = binaryUrl
      ? `Area putih = probabilitas >= threshold ${formatThreshold(usedThreshold)}. Area hitam = di bawah threshold.`
      : "Binary mask belum dibuat karena threshold tidak tersedia.";
  }
  renderMapLegend(payload.admin_regions || []);

  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadHeatmap.href = heatmapUrl || "#";
  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadOverlayPlain.href = overlayPlainUrl || "#";
  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadBinary.href = binaryUrl || "#";
  // Link unduhan diarahkan ke file hasil yang disimpan backend.
  downloadZip.href = zipUrl || "#";
  // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
  downloadHeatmap.classList.toggle("disabled", !heatmapUrl);
  updateRegionNumberOverlay();
  // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
  downloadOverlayPlain.classList.toggle("disabled", !overlayPlainUrl);
  // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
  downloadBinary.classList.toggle("disabled", !binaryUrl);
  // Class CSS diubah untuk menampilkan/menyembunyikan panel sesuai ketersediaan output.
  downloadZip.classList.toggle("disabled", !zipUrl);

  // `targetDateText` menyimpan tanggal input atau target H+1.
  const targetDateText = payload.target_date ? `Tanggal prediksi: ${payload.target_date}. ` : "";
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  resultMeta.textContent =
    `Prediksi H+${payload.horizon} selesai. ${targetDateText}Kode hasil: ${payload.prediction_id}. ` +
    `Model: ${modelProfile}. Threshold rekomendasi: ${formatThreshold(recommendedThreshold)}. ` +
    `Waktu proses: ${formatDuration(payload.processing_time_ms)}.`;

  // HTML panel/tabel dirender ulang dari payload backend.
  outputList.innerHTML = "";
  for (const output of payload.outputs || []) {
    const line = document.createElement("div");
    const links = [
      `<a href="${escapeHtml(output.heatmap_url)}" target="_blank" rel="noopener">heatmap</a>`,
    ];
    if (output.overlay_url) links.push(`<a href="${escapeHtml(output.overlay_url)}" target="_blank" rel="noopener">overlay wilayah</a>`);
    if (output.overlay_no_numbers_url) links.push(`<a href="${escapeHtml(output.overlay_no_numbers_url)}" target="_blank" rel="noopener">overlay tanpa nomor</a>`);
    if (output.overlay_plain_url) links.push(`<a href="${escapeHtml(output.overlay_plain_url)}" target="_blank" rel="noopener">overlay bersih</a>`);
    if (output.binary_url) links.push(`<a href="${escapeHtml(output.binary_url)}" target="_blank" rel="noopener">binary</a>`);
    // HTML panel/tabel dirender ulang dari payload backend.
    line.innerHTML = `<strong>H+${escapeHtml(output.step)}</strong>: ${links.join(" | ")}`;
    outputList.appendChild(line);
  }

  renderSpotlight(payload);
}

// Mengisi tabel riwayat prediksi dari metadata yang tersimpan di backend.
function renderHistory(items) {
  if (!Array.isArray(items) || items.length === 0) {
    // HTML panel/tabel dirender ulang dari payload backend.
    historyBody.innerHTML = `<tr><td colspan="7" class="text-secondary">Belum ada data.</td></tr>`;
    return;
  }

  // HTML panel/tabel dirender ulang dari payload backend.
  historyBody.innerHTML = "";
  for (const item of items) {
    const firstOutput = (item.outputs || [])[0] || {};
    const outputLinks = [];
    if (firstOutput.heatmap_url) {
      outputLinks.push(`<a href="${escapeHtml(firstOutput.heatmap_url)}" target="_blank" rel="noopener">Heatmap</a>`);
    }
    if (firstOutput.overlay_url) {
      outputLinks.push(`<a href="${escapeHtml(firstOutput.overlay_url)}" target="_blank" rel="noopener">Overlay</a>`);
    }
    if (firstOutput.overlay_no_numbers_url) {
      outputLinks.push(`<a href="${escapeHtml(firstOutput.overlay_no_numbers_url)}" target="_blank" rel="noopener">Overlay Tanpa Nomor</a>`);
    }
    if (firstOutput.overlay_plain_url) {
      outputLinks.push(`<a href="${escapeHtml(firstOutput.overlay_plain_url)}" target="_blank" rel="noopener">Overlay Bersih</a>`);
    }
    const outputCell = outputLinks.length ? outputLinks.join(" | ") : "-";

    const predictionId = item.prediction_id || "";
    const zipUrl = item.download_zip_url || `/api/predictions/${predictionId}/download`;
    const actions = predictionId
      ? `<button class="btn btn-sm btn-outline-primary history-detail-button" data-prediction-id="${escapeHtml(predictionId)}">Detail</button>
         <a class="btn btn-sm btn-outline-dark" href="${escapeHtml(zipUrl)}">ZIP</a>`
      : "-";

    const row = document.createElement("tr");
    // HTML panel/tabel dirender ulang dari payload backend.
    row.innerHTML = `
      <td><code>${escapeHtml(item.prediction_id || "-")}</code></td>
      <td>${escapeHtml(formatDate(item.created_at))}</td>
      <td>
        <div class="fw-semibold">H+${escapeHtml(item.horizon ?? "-")}</div>
        ${item.target_date ? `<div class="text-secondary">Target: ${escapeHtml(item.target_date)}</div>` : ""}
        <div class="text-secondary">${escapeHtml(item.model_profile || "-")}</div>
      </td>
      <td>${escapeHtml(formatThreshold(item.threshold))}</td>
      <td>${escapeHtml(formatDuration(item.processing_time_ms))}</td>
      <td>${outputCell}</td>
      <td class="d-flex gap-2 flex-wrap">${actions}</td>
    `;
    historyBody.appendChild(row);
  }
}

// Memanggil /api/predictions untuk mengambil riwayat prediksi terbaru.
async function loadHistory() {
  try {
    // Response ini berasal dari endpoint backend prediksi/status/riwayat.
    const response = await fetch("/api/predictions?limit=20");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(responseErrorText(payload));
    }
    renderHistory(payload);
    renderDashboardStats(payload);
  } catch (error) {
    // HTML panel/tabel dirender ulang dari payload backend.
    historyBody.innerHTML = `<tr><td colspan="7" class="text-danger">${escapeHtml(error.message)}</td></tr>`;
    setText(historySummary, "Gagal memuat ringkasan riwayat prediksi.");
    renderDashboardStats([]);
  }
}

// Memanggil /api/predictions/{id} untuk membuka kembali hasil prediksi lama.
async function loadPredictionDetail(predictionId) {
  hideError();
  try {
    // Response ini berasal dari endpoint backend prediksi/status/riwayat.
    const response = await fetch(`/api/predictions/${predictionId}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(responseErrorText(payload));
    }
    renderResult(payload, null);
  } catch (error) {
    showError(error.message || "Gagal memuat detail prediksi.");
  }
}

// Mengirim FormData berisi tujuh citra dan threshold ke /api/predict.
async function runPrediction(event) {
  event.preventDefault();
  hideError();

  // `files` menyimpan file upload atau nama file citra historis.
  const files = getSelectedFiles();
  // Cabang ini menangani validasi jumlah/urutan file upload citra historis.
  if (files.length !== 7) {
    showError("Input harus 7 file.");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    // Setiap file citra historis dimasukkan ke FormData dengan field `files` untuk FastAPI.
    formData.append("files", file, file.name);
  }
  // Cabang ini menangani kondisi threshold kosong, valid, atau perlu diformat.
  if (thresholdInput.value !== "") {
    // Threshold dari form dikirim agar backend membuat binary mask sesuai ambang pengguna.
    formData.append("threshold", thresholdInput.value);
  }
  formData.append("horizon", horizonInput.value);

  predictButton.disabled = true;
  // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
  predictButton.textContent = "Memproses...";
  try {
    // Response ini berasal dari endpoint backend prediksi/status/riwayat.
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(responseErrorText(payload));
    }
    renderResult(payload, files);
    await loadHistory();
  } catch (error) {
    showError(error.message || "Prediksi gagal.");
  } finally {
    // Teks dashboard diperbarui agar metadata prediksi tampil ke pengguna.
    predictButton.textContent = "Prediksi";
    validateSelection();
  }
}

// Menangani klik tombol detail pada tabel riwayat.
function onHistoryClick(event) {
  const button = event.target.closest(".history-detail-button");
  if (!button) return;
  const predictionId = button.dataset.predictionId;
  if (!predictionId) return;
  loadPredictionDetail(predictionId);
}

// Membuka detail prediksi yang sedang tampil di panel sorotan.
function onSpotlightOpen() {
  if (!spotlightPredictionId) return;
  loadPredictionDetail(spotlightPredictionId);
}

// Memuat ulang status runtime dan riwayat secara bersamaan.
async function refreshDashboard() {
  await Promise.all([loadRuntimeStatus(), loadHistory()]);
}

filesInput.addEventListener("change", validateSelection);
form.addEventListener("submit", runPrediction);
resetButton.addEventListener("click", resetFormState);
refreshHistoryButton.addEventListener("click", loadHistory);
refreshRuntimeButton.addEventListener("click", loadRuntimeStatus);
historyBody.addEventListener("click", onHistoryClick);
spotlightOpenButton.addEventListener("click", onSpotlightOpen);
if (regionNumberToggle) {
  regionNumberToggle.addEventListener("change", updateRegionNumberOverlay);
}

resetResultSection();
validateSelection();
refreshDashboard();
