// File anotasi dari `backend/static/app.js`.
// File ini untuk belajar/lampiran, bukan source utama aplikasi.
// Konteks laporan: Deployment, yaitu menjalankan model di sistem web.
// Setiap komentar menjelaskan tujuan baris kode di bawahnya secara singkat.
// Catatan asli pada script halaman web.
// Komentar file skripsi: Frontend dashboard: mengatur upload 7 citra H-6 sampai H0, memanggil API prediksi, menampilkan hasil, dan riwayat.
// Catatan asli pada script halaman web.
// Konteks laporan: mendukung proses eksperimen/training pada BAB IV.

// Menyimpan data tetap atau elemen halaman ke `appConfig`.
const appConfig = window.APP_CONFIG || {};
// Menyimpan data tetap atau elemen halaman ke `requiredStems`.
const requiredStems = appConfig.requiredStems || [];

// Menyimpan data tetap atau elemen halaman ke `form`.
const form = document.getElementById("predict-form");
// Menyimpan data tetap atau elemen halaman ke `filesInput`.
const filesInput = document.getElementById("files");
// Menyimpan data tetap atau elemen halaman ke `thresholdInput`.
const thresholdInput = document.getElementById("threshold");
// Menyimpan data tetap atau elemen halaman ke `horizonInput`.
const horizonInput = document.getElementById("horizon");
// Menyimpan data tetap atau elemen halaman ke `predictButton`.
const predictButton = document.getElementById("predict-button");
// Menyimpan data tetap atau elemen halaman ke `resetButton`.
const resetButton = document.getElementById("reset-button");
// Menyimpan data tetap atau elemen halaman ke `fileCounter`.
const fileCounter = document.getElementById("file-counter");
// Menyimpan data tetap atau elemen halaman ke `fileList`.
const fileList = document.getElementById("file-list");
// Menyimpan data tetap atau elemen halaman ke `errorBox`.
const errorBox = document.getElementById("error-box");

// Menyimpan data tetap atau elemen halaman ke `resultSection`.
const resultSection = document.getElementById("result-section");
// Menyimpan data tetap atau elemen halaman ke `resultInput`.
const resultInput = document.getElementById("result-input");
// Menyimpan data tetap atau elemen halaman ke `resultHeatmap`.
const resultHeatmap = document.getElementById("result-heatmap");
// Menyimpan data tetap atau elemen halaman ke `resultOverlay`.
const resultOverlay = document.getElementById("result-overlay");
// Menyimpan data tetap atau elemen halaman ke `resultMeta`.
const resultMeta = document.getElementById("result-meta");
// Menyimpan data tetap atau elemen halaman ke `outputList`.
const outputList = document.getElementById("output-list");
// Menyimpan data tetap atau elemen halaman ke `downloadHeatmap`.
const downloadHeatmap = document.getElementById("download-heatmap");
// Menyimpan data tetap atau elemen halaman ke `downloadOverlay`.
const downloadOverlay = document.getElementById("download-overlay");
// Menyimpan data tetap atau elemen halaman ke `downloadZip`.
const downloadZip = document.getElementById("download-zip");

// Menyimpan data tetap atau elemen halaman ke `historyBody`.
const historyBody = document.getElementById("history-body");
// Menyimpan data tetap atau elemen halaman ke `historySummary`.
const historySummary = document.getElementById("history-summary");
// Menyimpan data tetap atau elemen halaman ke `refreshHistoryButton`.
const refreshHistoryButton = document.getElementById("refresh-history");

// Menyimpan data tetap atau elemen halaman ke `refreshRuntimeButton`.
const refreshRuntimeButton = document.getElementById("refresh-runtime");
// Menyimpan data tetap atau elemen halaman ke `runtimeStateBadge`.
const runtimeStateBadge = document.getElementById("runtime-state-badge");
// Menyimpan data tetap atau elemen halaman ke `runtimeStateText`.
const runtimeStateText = document.getElementById("runtime-state-text");
// Menyimpan data tetap atau elemen halaman ke `runtimeModelProfile`.
const runtimeModelProfile = document.getElementById("runtime-model-profile");
// Menyimpan data tetap atau elemen halaman ke `runtimeModelName`.
const runtimeModelName = document.getElementById("runtime-model-name");
// Menyimpan data tetap atau elemen halaman ke `runtimeEngine`.
const runtimeEngine = document.getElementById("runtime-engine");
// Menyimpan data tetap atau elemen halaman ke `runtimePreprocess`.
const runtimePreprocess = document.getElementById("runtime-preprocess");
// Menyimpan data tetap atau elemen halaman ke `runtimeGrid`.
const runtimeGrid = document.getElementById("runtime-grid");
// Menyimpan data tetap atau elemen halaman ke `runtimePatch`.
const runtimePatch = document.getElementById("runtime-patch");
// Menyimpan data tetap atau elemen halaman ke `runtimeThreshold`.
const runtimeThreshold = document.getElementById("runtime-threshold");
// Menyimpan data tetap atau elemen halaman ke `runtimeTfVersion`.
const runtimeTfVersion = document.getElementById("runtime-tf-version");
// Menyimpan data tetap atau elemen halaman ke `runtimeCandidates`.
const runtimeCandidates = document.getElementById("runtime-candidates");

// Menyimpan data tetap atau elemen halaman ke `statTotalPredictions`.
const statTotalPredictions = document.getElementById("stat-total-predictions");
// Menyimpan data tetap atau elemen halaman ke `statTotalNote`.
const statTotalNote = document.getElementById("stat-total-note");
// Menyimpan data tetap atau elemen halaman ke `statLatestRun`.
const statLatestRun = document.getElementById("stat-latest-run");
// Menyimpan data tetap atau elemen halaman ke `statLatestNote`.
const statLatestNote = document.getElementById("stat-latest-note");
// Menyimpan data tetap atau elemen halaman ke `statAvgProcessing`.
const statAvgProcessing = document.getElementById("stat-avg-processing");
// Menyimpan data tetap atau elemen halaman ke `statAvgNote`.
const statAvgNote = document.getElementById("stat-avg-note");
// Menyimpan data tetap atau elemen halaman ke `statActiveEngine`.
const statActiveEngine = document.getElementById("stat-active-engine");
// Menyimpan data tetap atau elemen halaman ke `statEngineNote`.
const statEngineNote = document.getElementById("stat-engine-note");

// Menyimpan data tetap atau elemen halaman ke `spotlightEmpty`.
const spotlightEmpty = document.getElementById("spotlight-empty");
// Menyimpan data tetap atau elemen halaman ke `spotlightContent`.
const spotlightContent = document.getElementById("spotlight-content");
// Menyimpan data tetap atau elemen halaman ke `spotlightHeatmap`.
const spotlightHeatmap = document.getElementById("spotlight-heatmap");
// Menyimpan data tetap atau elemen halaman ke `spotlightOverlay`.
const spotlightOverlay = document.getElementById("spotlight-overlay");
// Menyimpan data tetap atau elemen halaman ke `spotlightTitle`.
const spotlightTitle = document.getElementById("spotlight-title");
// Menyimpan data tetap atau elemen halaman ke `spotlightMeta`.
const spotlightMeta = document.getElementById("spotlight-meta");
// Menyimpan data tetap atau elemen halaman ke `spotlightOpenButton`.
const spotlightOpenButton = document.getElementById("spotlight-open");

// Menyimpan data sementara ke `currentRuntimeStatus` yang bisa berubah saat halaman dipakai.
let currentRuntimeStatus = null;
// Menyimpan data sementara ke `spotlightPredictionId` yang bisa berubah saat halaman dipakai.
let spotlightPredictionId = null;

// Membuat langkah kerja `normalizeStem` untuk interaksi halaman.
function normalizeStem(name) {
  // Menyimpan data tetap atau elemen halaman ke `base`.
  const base = name.split(".").slice(0, -1).join(".").trim().toUpperCase().replace(/\s+/g, "");
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (base === "H-0") return "H0";
  // Mengirim hasil dari langkah kerja JavaScript.
  return base;
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `getSelectedFiles` untuk interaksi halaman.
function getSelectedFiles() {
  // Mengirim hasil dari langkah kerja JavaScript.
  return Array.from(filesInput.files || []);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `escapeHtml` untuk interaksi halaman.
function escapeHtml(value) {
  // Mengirim hasil dari langkah kerja JavaScript.
  return String(value ?? "")
    // Melanjutkan proses interaksi pada dashboard web.
    .replace(/&/g, "&amp;")
    // Melanjutkan proses interaksi pada dashboard web.
    .replace(/</g, "&lt;")
    // Melanjutkan proses interaksi pada dashboard web.
    .replace(/>/g, "&gt;")
    // Melanjutkan proses interaksi pada dashboard web.
    .replace(/"/g, "&quot;")
    // Melanjutkan proses interaksi pada dashboard web.
    .replace(/'/g, "&#39;");
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `setText` untuk interaksi halaman.
function setText(node, value, fallback = "-") {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!node) return;
  // Menyimpan data tetap atau elemen halaman ke `resolved`.
  const resolved = value === null || value === undefined || value === "" ? fallback : value;
  // Mengubah tulisan yang terlihat pada dashboard.
  node.textContent = String(resolved);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `setImageSource` untuk interaksi halaman.
function setImageSource(node, url) {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!node) return;
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (url) {
    // Mengatur nilai `node.src` untuk memperbarui data atau tampilan halaman.
    node.src = url;
  // Melanjutkan proses interaksi pada dashboard web.
  } else {
    // Melanjutkan proses interaksi pada dashboard web.
    node.removeAttribute("src");
  // Melanjutkan proses interaksi pada dashboard web.
  }
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `hideError` untuk interaksi halaman.
function hideError() {
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  errorBox.classList.add("d-none");
  // Mengubah tulisan yang terlihat pada dashboard.
  errorBox.textContent = "";
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `showError` untuk interaksi halaman.
function showError(message) {
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  errorBox.classList.remove("d-none");
  // Mengubah tulisan yang terlihat pada dashboard.
  errorBox.textContent = message;
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `formatDate` untuk interaksi halaman.
function formatDate(value) {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!value) return "-";
  // Menyimpan data tetap atau elemen halaman ke `date`.
  const date = new Date(value);
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (Number.isNaN(date.getTime())) return value;
  // Mengirim hasil dari langkah kerja JavaScript.
  return date.toLocaleString("id-ID", {
    // Melanjutkan proses interaksi pada dashboard web.
    year: "numeric",
    // Melanjutkan proses interaksi pada dashboard web.
    month: "short",
    // Melanjutkan proses interaksi pada dashboard web.
    day: "2-digit",
    // Melanjutkan proses interaksi pada dashboard web.
    hour: "2-digit",
    // Melanjutkan proses interaksi pada dashboard web.
    minute: "2-digit",
    // Melanjutkan proses interaksi pada dashboard web.
    second: "2-digit",
    // Melanjutkan proses interaksi pada dashboard web.
    hour12: false,
  // Melanjutkan proses interaksi pada dashboard web.
  });
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `formatDuration` untuk interaksi halaman.
function formatDuration(ms) {
  // Menyimpan data tetap atau elemen halaman ke `numeric`.
  const numeric = Number(ms);
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!Number.isFinite(numeric)) return "-";
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (numeric >= 1000) {
    // Mengirim hasil dari langkah kerja JavaScript.
    return `${(numeric / 1000).toFixed(1)} dtk`;
  // Melanjutkan proses interaksi pada dashboard web.
  }
  // Mengirim hasil dari langkah kerja JavaScript.
  return `${Math.round(numeric)} ms`;
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `formatThreshold` untuk interaksi halaman.
function formatThreshold(value) {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (value === null || value === undefined || value === "") return "-";
  // Menyimpan data tetap atau elemen halaman ke `numeric`.
  const numeric = Number(value);
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!Number.isFinite(numeric)) return String(value);
  // Mengirim hasil dari langkah kerja JavaScript.
  return numeric.toFixed(2);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `summarizePatchInfo` untuk interaksi halaman.
function summarizePatchInfo(payload) {
  // Menyimpan data tetap atau elemen halaman ke `patchSize`.
  const patchSize = payload?.patch_size;
  // Menyimpan data tetap atau elemen halaman ke `patchStride`.
  const patchStride = payload?.patch_stride;
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!patchSize && !patchStride) return "Tidak digunakan";
  // Menyimpan data tetap atau elemen halaman ke `sizeText`.
  const sizeText = patchSize ? `${patchSize} px` : "-";
  // Menyimpan data tetap atau elemen halaman ke `strideText`.
  const strideText = patchStride ? `${patchStride} px` : "-";
  // Mengirim hasil dari langkah kerja JavaScript.
  return `${sizeText} / ${strideText}`;
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `summarizeGridInfo` untuk interaksi halaman.
function summarizeGridInfo(payload) {
  // Menyimpan data tetap atau elemen halaman ke `gridSize`.
  const gridSize = payload?.grid_size ?? appConfig.gridSize;
  // Menyimpan data tetap atau elemen halaman ke `channels`.
  const channels = payload?.channels ?? "-";
  // Menyimpan data tetap atau elemen halaman ke `timeSteps`.
  const timeSteps = payload?.time_steps ?? requiredStems.length;
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!gridSize) return "-";
  // Mengirim hasil dari langkah kerja JavaScript.
  return `${gridSize} x ${gridSize} | ${channels} ch | ${timeSteps} frame`;
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `summarizeCandidates` untuk interaksi halaman.
function summarizeCandidates(candidates) {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!Array.isArray(candidates) || candidates.length === 0) return "Tidak ada kandidat model yang terdeteksi.";
  // Mengirim hasil dari langkah kerja JavaScript.
  return candidates
    // Melanjutkan proses interaksi pada dashboard web.
    .slice(0, 4)
    // Mengatur nilai `.map((item)` untuk memperbarui data atau tampilan halaman.
    .map((item) => String(item).split(/[\\/]/).pop())
    // Melanjutkan proses interaksi pada dashboard web.
    .join(" | ");
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `responseErrorText` untuk interaksi halaman.
function responseErrorText(payload) {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!payload) return "Terjadi kesalahan pada server.";
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (typeof payload.detail === "string") return payload.detail;
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (Array.isArray(payload.detail)) {
    // Mengirim hasil dari langkah kerja JavaScript.
    return payload.detail.map((item) => item.msg || JSON.stringify(item)).join(", ");
  // Melanjutkan proses interaksi pada dashboard web.
  }
  // Mengirim hasil dari langkah kerja JavaScript.
  return JSON.stringify(payload);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `resetResultSection` untuk interaksi halaman.
function resetResultSection() {
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  resultSection.classList.add("d-none");
  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(resultInput, "");
  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(resultHeatmap, "");
  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(resultOverlay, "");
  // Mengubah tulisan yang terlihat pada dashboard.
  resultMeta.textContent = "";
  // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
  outputList.innerHTML = "";

  // Mengatur nilai `downloadHeatmap.href` untuk memperbarui data atau tampilan halaman.
  downloadHeatmap.href = "#";
  // Mengatur nilai `downloadOverlay.href` untuk memperbarui data atau tampilan halaman.
  downloadOverlay.href = "#";
  // Mengatur nilai `downloadZip.href` untuk memperbarui data atau tampilan halaman.
  downloadZip.href = "#";

  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  downloadHeatmap.classList.add("disabled");
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  downloadOverlay.classList.add("disabled");
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  downloadZip.classList.add("disabled");
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `resetFormState` untuk interaksi halaman.
function resetFormState() {
  // Melanjutkan proses interaksi pada dashboard web.
  form.reset();
  // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
  fileList.innerHTML = "";
  // Mengubah tulisan yang terlihat pada dashboard.
  fileCounter.textContent = "0/7 file terunggah";
  // Melanjutkan proses interaksi pada dashboard web.
  hideError();
  // Melanjutkan proses interaksi pada dashboard web.
  resetResultSection();
  // Melanjutkan proses interaksi pada dashboard web.
  validateSelection();
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `validateSelection` untuk interaksi halaman.
function validateSelection() {
  // Melanjutkan proses interaksi pada dashboard web.
  hideError();
  // Menyimpan data tetap atau elemen halaman ke `files`.
  const files = getSelectedFiles();
  // Menyimpan data tetap atau elemen halaman ke `stems`.
  const stems = files.map((f) => normalizeStem(f.name));
  // Menyimpan data tetap atau elemen halaman ke `uniqueStems`.
  const uniqueStems = new Set(stems);

  // Mengubah tulisan yang terlihat pada dashboard.
  fileCounter.textContent = `${files.length}/7 file terunggah`;
  // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
  fileList.innerHTML = "";
  // Mengulang proses untuk setiap data yang ditampilkan di halaman.
  for (const file of files) {
    // Menyimpan data tetap atau elemen halaman ke `item`.
    const item = document.createElement("li");
    // Mengubah tulisan yang terlihat pada dashboard.
    item.textContent = file.name;
    // Melanjutkan proses interaksi pada dashboard web.
    fileList.appendChild(item);
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Mengecek syarat sebelum halaman menjalankan proses.
  if (files.length !== 7) {
    // Mengatur nilai `predictButton.disabled` untuk memperbarui data atau tampilan halaman.
    predictButton.disabled = true;
    // Melanjutkan proses interaksi pada dashboard web.
    return;
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Mengecek syarat sebelum halaman menjalankan proses.
  if (uniqueStems.size !== 7) {
    // Melanjutkan proses interaksi pada dashboard web.
    showError("Terdapat nama file yang duplikat.");
    // Mengatur nilai `predictButton.disabled` untuk memperbarui data atau tampilan halaman.
    predictButton.disabled = true;
    // Melanjutkan proses interaksi pada dashboard web.
    return;
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Mengulang proses untuk setiap data yang ditampilkan di halaman.
  for (const expected of requiredStems) {
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (!uniqueStems.has(expected)) {
      // Melanjutkan proses interaksi pada dashboard web.
      showError(`Nama file tidak lengkap. Pastikan ada file ${expected}.`);
      // Mengatur nilai `predictButton.disabled` untuk memperbarui data atau tampilan halaman.
      predictButton.disabled = true;
      // Melanjutkan proses interaksi pada dashboard web.
      return;
    // Melanjutkan proses interaksi pada dashboard web.
    }
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Mengatur nilai `predictButton.disabled` untuk memperbarui data atau tampilan halaman.
  predictButton.disabled = false;
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `renderRuntimeStatus` untuk interaksi halaman.
function renderRuntimeStatus(payload) {
  // Mengatur nilai `currentRuntimeStatus` untuk memperbarui data atau tampilan halaman.
  currentRuntimeStatus = payload;

  // Menyimpan data tetap atau elemen halaman ke `engineName`.
  const engineName = payload?.prediction_engine || payload?.model_backend || "-";
  // Menyimpan data tetap atau elemen halaman ke `usingFallback`.
  const usingFallback = String(engineName).toLowerCase().includes("heuristic");

  // Mengatur nilai `runtimeStateBadge.className` untuk memperbarui data atau tampilan halaman.
  runtimeStateBadge.className = `status-badge ${usingFallback ? "status-warning" : "status-success"}`;
  // Mengubah tulisan yang terlihat pada dashboard.
  runtimeStateBadge.textContent = usingFallback ? "Fallback Heuristik" : "ConvLSTM Aktif";
  // Mengubah tulisan yang terlihat pada dashboard.
  runtimeStateText.textContent = usingFallback
    // Melanjutkan proses interaksi pada dashboard web.
    ? "Aplikasi tetap berjalan, tetapi inferensi saat ini menggunakan backend fallback."
    // Melanjutkan proses interaksi pada dashboard web.
    : "Model ConvLSTM berhasil dimuat dan siap dipakai untuk prediksi.";

  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeModelProfile, payload?.model_profile || appConfig.modelProfile);
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeModelName, payload?.model_display_name || appConfig.modelDisplayName);
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeEngine, engineName);
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimePreprocess, payload?.preprocess_mode);
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeGrid, summarizeGridInfo(payload));
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimePatch, summarizePatchInfo(payload));
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeThreshold, formatThreshold(payload?.recommended_threshold));
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeTfVersion, payload?.tensorflow_version || "Tidak tersedia");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeCandidates, summarizeCandidates(payload?.model_candidates));

  // Melanjutkan proses interaksi pada dashboard web.
  setText(statActiveEngine, engineName);
  // Melanjutkan proses interaksi pada dashboard web.
  setText(
    // Melanjutkan proses interaksi pada dashboard web.
    statEngineNote,
    // Melanjutkan proses interaksi pada dashboard web.
    `${payload?.model_profile || appConfig.modelProfile || "-"} | ${payload?.inference_mode || appConfig.inferenceMode || "-"}`,
  // Melanjutkan proses interaksi pada dashboard web.
  );
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `renderRuntimeError` untuk interaksi halaman.
function renderRuntimeError(message) {
  // Mengatur nilai `runtimeStateBadge.className` untuk memperbarui data atau tampilan halaman.
  runtimeStateBadge.className = "status-badge status-danger";
  // Mengubah tulisan yang terlihat pada dashboard.
  runtimeStateBadge.textContent = "Gagal Memuat";
  // Mengubah tulisan yang terlihat pada dashboard.
  runtimeStateText.textContent = message;

  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeModelProfile, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeModelName, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeEngine, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimePreprocess, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeGrid, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimePatch, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeThreshold, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeTfVersion, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(runtimeCandidates, "-");

  // Melanjutkan proses interaksi pada dashboard web.
  setText(statActiveEngine, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(statEngineNote, "Status runtime belum tersedia");
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `loadRuntimeStatus` yang bisa menunggu jawaban dari backend.
async function loadRuntimeStatus() {
  // Melanjutkan proses interaksi pada dashboard web.
  try {
    // Menyimpan data tetap atau elemen halaman ke `response`.
    const response = await fetch("/api/runtime/status");
    // Menyimpan data tetap atau elemen halaman ke `payload`.
    const payload = await response.json();
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (!response.ok) {
      // Melanjutkan proses interaksi pada dashboard web.
      throw new Error(responseErrorText(payload));
    // Melanjutkan proses interaksi pada dashboard web.
    }
    // Melanjutkan proses interaksi pada dashboard web.
    renderRuntimeStatus(payload);
  // Melanjutkan proses interaksi pada dashboard web.
  } catch (error) {
    // Melanjutkan proses interaksi pada dashboard web.
    renderRuntimeError(error.message || "Gagal mengambil status runtime.");
  // Melanjutkan proses interaksi pada dashboard web.
  }
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `renderSpotlightEmpty` untuk interaksi halaman.
function renderSpotlightEmpty() {
  // Mengatur nilai `spotlightPredictionId` untuk memperbarui data atau tampilan halaman.
  spotlightPredictionId = null;
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  spotlightEmpty.classList.remove("d-none");
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  spotlightContent.classList.add("d-none");
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  spotlightOpenButton.classList.add("d-none");
  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(spotlightHeatmap, "");
  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(spotlightOverlay, "");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(spotlightTitle, "-");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(spotlightMeta, "Belum ada hasil yang dapat disorot.");
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `spotlightMetaText` untuk interaksi halaman.
function spotlightMetaText(payload) {
  // Menyimpan data tetap atau elemen halaman ke `parts`.
  const parts = [
    // Melanjutkan proses interaksi pada dashboard web.
    `Waktu: ${formatDate(payload.created_at)}`,
    // Melanjutkan proses interaksi pada dashboard web.
    `Model: ${payload.model_profile || "-"}`,
    // Melanjutkan proses interaksi pada dashboard web.
    `Backend: ${payload.prediction_engine || payload.model_backend || "-"}`,
    // Melanjutkan proses interaksi pada dashboard web.
    `Horizon: H+${payload.horizon ?? "-"}`,
    // Melanjutkan proses interaksi pada dashboard web.
    `Threshold: ${formatThreshold(payload.threshold)}`,
    // Melanjutkan proses interaksi pada dashboard web.
    `Proses: ${formatDuration(payload.processing_time_ms)}`,
  // Melanjutkan proses interaksi pada dashboard web.
  ];
  // Mengirim hasil dari langkah kerja JavaScript.
  return parts.join(" | ");
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `renderSpotlight` untuk interaksi halaman.
function renderSpotlight(payload) {
  // Menyimpan data tetap atau elemen halaman ke `firstOutput`.
  const firstOutput = (payload?.outputs || [])[0] || {};
  // Menyimpan data tetap atau elemen halaman ke `stamp`.
  const stamp = `t=${Date.now()}`;
  // Menyimpan data tetap atau elemen halaman ke `heatmapUrl`.
  const heatmapUrl = firstOutput.heatmap_url || payload?.heatmap_url || "";
  // Menyimpan data tetap atau elemen halaman ke `overlayUrl`.
  const overlayUrl = firstOutput.overlay_url || payload?.overlay_url || "";
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!heatmapUrl && !overlayUrl) {
    // Melanjutkan proses interaksi pada dashboard web.
    renderSpotlightEmpty();
    // Melanjutkan proses interaksi pada dashboard web.
    return;
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Mengatur nilai `spotlightPredictionId` untuk memperbarui data atau tampilan halaman.
  spotlightPredictionId = payload?.prediction_id || null;
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  spotlightEmpty.classList.add("d-none");
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  spotlightContent.classList.remove("d-none");
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  spotlightOpenButton.classList.toggle("d-none", !spotlightPredictionId);

  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(spotlightHeatmap, heatmapUrl ? `${heatmapUrl}?${stamp}` : "");
  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(spotlightOverlay, overlayUrl ? `${overlayUrl}?${stamp}` : "");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(spotlightTitle, `Prediksi ${payload?.prediction_id || "-"}`);
  // Melanjutkan proses interaksi pada dashboard web.
  setText(spotlightMeta, spotlightMetaText(payload));
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `renderDashboardStats` untuk interaksi halaman.
function renderDashboardStats(items) {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!Array.isArray(items) || items.length === 0) {
    // Melanjutkan proses interaksi pada dashboard web.
    setText(statTotalPredictions, "0");
    // Melanjutkan proses interaksi pada dashboard web.
    setText(statTotalNote, "Belum ada riwayat tersimpan");
    // Melanjutkan proses interaksi pada dashboard web.
    setText(statLatestRun, "-");
    // Melanjutkan proses interaksi pada dashboard web.
    setText(statLatestNote, "Menunggu prediksi pertama");
    // Melanjutkan proses interaksi pada dashboard web.
    setText(statAvgProcessing, "-");
    // Melanjutkan proses interaksi pada dashboard web.
    setText(statAvgNote, "Belum ada data waktu proses");
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (!currentRuntimeStatus) {
      // Melanjutkan proses interaksi pada dashboard web.
      setText(statActiveEngine, "-");
      // Melanjutkan proses interaksi pada dashboard web.
      setText(statEngineNote, "Status runtime belum tersedia");
    // Melanjutkan proses interaksi pada dashboard web.
    }
    // Melanjutkan proses interaksi pada dashboard web.
    setText(historySummary, "Belum ada riwayat prediksi yang tersimpan.");
    // Melanjutkan proses interaksi pada dashboard web.
    renderSpotlightEmpty();
    // Melanjutkan proses interaksi pada dashboard web.
    return;
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Menyimpan data tetap atau elemen halaman ke `latest`.
  const latest = items[0];
  // Menyimpan data tetap atau elemen halaman ke `processingValues`.
  const processingValues = items
    // Mengatur nilai `.map((item)` untuk memperbarui data atau tampilan halaman.
    .map((item) => Number(item.processing_time_ms))
    // Mengatur nilai `.filter((value)` untuk memperbarui data atau tampilan halaman.
    .filter((value) => Number.isFinite(value));
  // Menyimpan data tetap atau elemen halaman ke `averageMs`.
  const averageMs = processingValues.length
    // Mengatur nilai `? Math.round(processingValues.reduce((sum, value)` untuk memperbarui data atau tampilan halaman.
    ? Math.round(processingValues.reduce((sum, value) => sum + value, 0) / processingValues.length)
    // Melanjutkan proses interaksi pada dashboard web.
    : null;

  // Melanjutkan proses interaksi pada dashboard web.
  setText(statTotalPredictions, String(items.length));
  // Melanjutkan proses interaksi pada dashboard web.
  setText(statTotalNote, "Riwayat terbaru yang berhasil dimuat");
  // Melanjutkan proses interaksi pada dashboard web.
  setText(statLatestRun, formatDate(latest.created_at));
  // Melanjutkan proses interaksi pada dashboard web.
  setText(statLatestNote, `${latest.model_profile || "-"} | H+${latest.horizon ?? "-"}`);
  // Melanjutkan proses interaksi pada dashboard web.
  setText(statAvgProcessing, formatDuration(averageMs));
  // Melanjutkan proses interaksi pada dashboard web.
  setText(statAvgNote, `${processingValues.length || 0} data waktu proses tersedia`);

  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!currentRuntimeStatus) {
    // Melanjutkan proses interaksi pada dashboard web.
    setText(statActiveEngine, latest.prediction_engine || latest.model_backend || "-");
    // Melanjutkan proses interaksi pada dashboard web.
    setText(statEngineNote, "Mengikuti riwayat prediksi terbaru");
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Melanjutkan proses interaksi pada dashboard web.
  setText(
    // Melanjutkan proses interaksi pada dashboard web.
    historySummary,
    // Melanjutkan proses interaksi pada dashboard web.
    `Menampilkan ${items.length} riwayat terbaru. Prediksi terakhir dibuat pada ${formatDate(latest.created_at)}.`,
  // Melanjutkan proses interaksi pada dashboard web.
  );
  // Melanjutkan proses interaksi pada dashboard web.
  renderSpotlight(latest);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `renderResult` untuk interaksi halaman.
function renderResult(payload, files = null) {
  // Menyimpan data tetap atau elemen halaman ke `stamp`.
  const stamp = `t=${Date.now()}`;
  // Menyimpan data tetap atau elemen halaman ke `selectedFiles`.
  const selectedFiles = Array.isArray(files) ? files : [];
  // Menyimpan data tetap atau elemen halaman ke `h0File`.
  const h0File = selectedFiles.find((f) => normalizeStem(f.name) === "H0");
  // Menyimpan data tetap atau elemen halaman ke `firstOutput`.
  const firstOutput = (payload.outputs || [])[0] || {};
  // Menyimpan data tetap atau elemen halaman ke `heatmapUrl`.
  const heatmapUrl = payload.heatmap_url || firstOutput.heatmap_url || "";
  // Menyimpan data tetap atau elemen halaman ke `overlayUrl`.
  const overlayUrl = payload.overlay_url || firstOutput.overlay_url || "";
  // Menyimpan data tetap atau elemen halaman ke `zipUrl`.
  const zipUrl = payload.download_zip_url || `/api/predictions/${payload.prediction_id}/download`;
  // Menyimpan data tetap atau elemen halaman ke `engineName`.
  const engineName = payload.prediction_engine || payload.model_backend || "-";
  // Menyimpan data tetap atau elemen halaman ke `preprocessMode`.
  const preprocessMode = payload.preprocess_mode || "-";
  // Menyimpan data tetap atau elemen halaman ke `modelProfile`.
  const modelProfile = payload.model_profile || appConfig.modelProfile || "-";
  // Menyimpan data tetap atau elemen halaman ke `inferenceMode`.
  const inferenceMode = payload.inference_mode || appConfig.inferenceMode || "-";
  // Menyimpan data tetap atau elemen halaman ke `recommendedThreshold`.
  const recommendedThreshold = payload.recommended_threshold ?? appConfig.recommendedThreshold ?? "-";
  // Menyimpan data tetap atau elemen halaman ke `outputShape`.
  const outputShape = Array.isArray(payload.output_shape) ? payload.output_shape.join("x") : "-";

  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  resultSection.classList.remove("d-none");
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (h0File) {
    // Mengatur nilai `resultInput.src` untuk memperbarui data atau tampilan halaman.
    resultInput.src = URL.createObjectURL(h0File);
  // Melanjutkan proses interaksi pada dashboard web.
  } else if (payload.input_urls?.H0) {
    // Mengatur nilai `resultInput.src` untuk memperbarui data atau tampilan halaman.
    resultInput.src = `${payload.input_urls.H0}?${stamp}`;
  // Melanjutkan proses interaksi pada dashboard web.
  } else {
    // Melanjutkan proses interaksi pada dashboard web.
    resultInput.removeAttribute("src");
  // Melanjutkan proses interaksi pada dashboard web.
  }
  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(resultHeatmap, heatmapUrl ? `${heatmapUrl}?${stamp}` : "");
  // Melanjutkan proses interaksi pada dashboard web.
  setImageSource(resultOverlay, overlayUrl ? `${overlayUrl}?${stamp}` : "");

  // Mengatur nilai `downloadHeatmap.href` untuk memperbarui data atau tampilan halaman.
  downloadHeatmap.href = heatmapUrl || "#";
  // Mengatur nilai `downloadOverlay.href` untuk memperbarui data atau tampilan halaman.
  downloadOverlay.href = overlayUrl || "#";
  // Mengatur nilai `downloadZip.href` untuk memperbarui data atau tampilan halaman.
  downloadZip.href = zipUrl || "#";
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  downloadHeatmap.classList.toggle("disabled", !heatmapUrl);
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  downloadOverlay.classList.toggle("disabled", !overlayUrl);
  // Mengubah gaya tampilan elemen, misalnya aktif, tersembunyi, atau error.
  downloadZip.classList.toggle("disabled", !zipUrl);

  // Mengubah tulisan yang terlihat pada dashboard.
  resultMeta.textContent =
    // Melanjutkan proses interaksi pada dashboard web.
    `ID: ${payload.prediction_id} | Model: ${modelProfile} | Engine: ${engineName} | ` +
    // Melanjutkan proses interaksi pada dashboard web.
    `Inference: ${inferenceMode} | Output: ${outputShape} | ` +
    // Melanjutkan proses interaksi pada dashboard web.
    `Horizon: ${payload.horizon} | Grid: ${payload.grid_size} | ` +
    // Melanjutkan proses interaksi pada dashboard web.
    `Preprocess: ${preprocessMode} | Threshold: ${payload.threshold ?? "-"} | ` +
    // Melanjutkan proses interaksi pada dashboard web.
    `Threshold rekomendasi: ${recommendedThreshold} | ` +
    // Melanjutkan proses interaksi pada dashboard web.
    `Proses: ${payload.processing_time_ms} ms`;

  // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
  outputList.innerHTML = "";
  // Mengulang proses untuk setiap data yang ditampilkan di halaman.
  for (const output of payload.outputs || []) {
    // Menyimpan data tetap atau elemen halaman ke `line`.
    const line = document.createElement("div");
    // Menyimpan data tetap atau elemen halaman ke `links`.
    const links = [
      // Mengatur nilai ``<a href` untuk memperbarui data atau tampilan halaman.
      `<a href="${escapeHtml(output.heatmap_url)}" target="_blank" rel="noopener">heatmap</a>`,
    // Melanjutkan proses interaksi pada dashboard web.
    ];
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (output.overlay_url) links.push(`<a href="${escapeHtml(output.overlay_url)}" target="_blank" rel="noopener">overlay</a>`);
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (output.binary_url) links.push(`<a href="${escapeHtml(output.binary_url)}" target="_blank" rel="noopener">binary</a>`);
    // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
    line.innerHTML = `<strong>H+${escapeHtml(output.step)}</strong>: ${links.join(" | ")}`;
    // Melanjutkan proses interaksi pada dashboard web.
    outputList.appendChild(line);
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Melanjutkan proses interaksi pada dashboard web.
  renderSpotlight(payload);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `renderHistory` untuk interaksi halaman.
function renderHistory(items) {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!Array.isArray(items) || items.length === 0) {
    // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
    historyBody.innerHTML = `<tr><td colspan="7" class="text-secondary">Belum ada data.</td></tr>`;
    // Melanjutkan proses interaksi pada dashboard web.
    return;
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
  historyBody.innerHTML = "";
  // Mengulang proses untuk setiap data yang ditampilkan di halaman.
  for (const item of items) {
    // Menyimpan data tetap atau elemen halaman ke `firstOutput`.
    const firstOutput = (item.outputs || [])[0] || {};
    // Menyimpan data tetap atau elemen halaman ke `outputLinks`.
    const outputLinks = [];
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (firstOutput.heatmap_url) {
      // Mengatur nilai `outputLinks.push(`<a href` untuk memperbarui data atau tampilan halaman.
      outputLinks.push(`<a href="${escapeHtml(firstOutput.heatmap_url)}" target="_blank" rel="noopener">Heatmap</a>`);
    // Melanjutkan proses interaksi pada dashboard web.
    }
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (firstOutput.overlay_url) {
      // Mengatur nilai `outputLinks.push(`<a href` untuk memperbarui data atau tampilan halaman.
      outputLinks.push(`<a href="${escapeHtml(firstOutput.overlay_url)}" target="_blank" rel="noopener">Overlay</a>`);
    // Melanjutkan proses interaksi pada dashboard web.
    }
    // Menyimpan data tetap atau elemen halaman ke `outputCell`.
    const outputCell = outputLinks.length ? outputLinks.join(" | ") : "-";

    // Menyimpan data tetap atau elemen halaman ke `predictionId`.
    const predictionId = item.prediction_id || "";
    // Menyimpan data tetap atau elemen halaman ke `zipUrl`.
    const zipUrl = item.download_zip_url || `/api/predictions/${predictionId}/download`;
    // Menyimpan data tetap atau elemen halaman ke `actions`.
    const actions = predictionId
      // Mengatur nilai `? `<button class` untuk memperbarui data atau tampilan halaman.
      ? `<button class="btn btn-sm btn-outline-primary history-detail-button" data-prediction-id="${escapeHtml(predictionId)}">Detail</button>
         // Mengatur nilai `<a class` untuk memperbarui data atau tampilan halaman.
         <a class="btn btn-sm btn-outline-dark" href="${escapeHtml(zipUrl)}">ZIP</a>`
      // Melanjutkan proses interaksi pada dashboard web.
      : "-";

    // Menyimpan data tetap atau elemen halaman ke `row`.
    const row = document.createElement("tr");
    // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
    row.innerHTML = `
      // Melanjutkan proses interaksi pada dashboard web.
      <td><code>${escapeHtml(item.prediction_id || "-")}</code></td>
      // Melanjutkan proses interaksi pada dashboard web.
      <td>${escapeHtml(formatDate(item.created_at))}</td>
      // Melanjutkan proses interaksi pada dashboard web.
      <td>
        // Mengatur nilai `<div class` untuk memperbarui data atau tampilan halaman.
        <div class="fw-semibold">H+${escapeHtml(item.horizon ?? "-")}</div>
        // Mengatur nilai `<div class` untuk memperbarui data atau tampilan halaman.
        <div class="text-secondary">${escapeHtml(item.model_profile || "-")}</div>
      // Melanjutkan proses interaksi pada dashboard web.
      </td>
      // Melanjutkan proses interaksi pada dashboard web.
      <td>${escapeHtml(formatThreshold(item.threshold))}</td>
      // Melanjutkan proses interaksi pada dashboard web.
      <td>${escapeHtml(formatDuration(item.processing_time_ms))}</td>
      // Melanjutkan proses interaksi pada dashboard web.
      <td>${outputCell}</td>
      // Mengatur nilai `<td class` untuk memperbarui data atau tampilan halaman.
      <td class="d-flex gap-2 flex-wrap">${actions}</td>
    // Melanjutkan proses interaksi pada dashboard web.
    `;
    // Melanjutkan proses interaksi pada dashboard web.
    historyBody.appendChild(row);
  // Melanjutkan proses interaksi pada dashboard web.
  }
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `loadHistory` yang bisa menunggu jawaban dari backend.
async function loadHistory() {
  // Melanjutkan proses interaksi pada dashboard web.
  try {
    // Menyimpan data tetap atau elemen halaman ke `response`.
    const response = await fetch("/api/predictions?limit=20");
    // Menyimpan data tetap atau elemen halaman ke `payload`.
    const payload = await response.json();
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (!response.ok) {
      // Melanjutkan proses interaksi pada dashboard web.
      throw new Error(responseErrorText(payload));
    // Melanjutkan proses interaksi pada dashboard web.
    }
    // Melanjutkan proses interaksi pada dashboard web.
    renderHistory(payload);
    // Melanjutkan proses interaksi pada dashboard web.
    renderDashboardStats(payload);
  // Melanjutkan proses interaksi pada dashboard web.
  } catch (error) {
    // Mengganti isi tampilan agar hasil terbaru muncul di halaman.
    historyBody.innerHTML = `<tr><td colspan="7" class="text-danger">${escapeHtml(error.message)}</td></tr>`;
    // Melanjutkan proses interaksi pada dashboard web.
    setText(historySummary, "Gagal memuat ringkasan riwayat prediksi.");
    // Melanjutkan proses interaksi pada dashboard web.
    renderDashboardStats([]);
  // Melanjutkan proses interaksi pada dashboard web.
  }
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `loadPredictionDetail` yang bisa menunggu jawaban dari backend.
async function loadPredictionDetail(predictionId) {
  // Melanjutkan proses interaksi pada dashboard web.
  hideError();
  // Melanjutkan proses interaksi pada dashboard web.
  try {
    // Menyimpan data tetap atau elemen halaman ke `response`.
    const response = await fetch(`/api/predictions/${predictionId}`);
    // Menyimpan data tetap atau elemen halaman ke `payload`.
    const payload = await response.json();
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (!response.ok) {
      // Melanjutkan proses interaksi pada dashboard web.
      throw new Error(responseErrorText(payload));
    // Melanjutkan proses interaksi pada dashboard web.
    }
    // Melanjutkan proses interaksi pada dashboard web.
    renderResult(payload, null);
  // Melanjutkan proses interaksi pada dashboard web.
  } catch (error) {
    // Melanjutkan proses interaksi pada dashboard web.
    showError(error.message || "Gagal memuat detail prediksi.");
  // Melanjutkan proses interaksi pada dashboard web.
  }
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `runPrediction` yang bisa menunggu jawaban dari backend.
async function runPrediction(event) {
  // Melanjutkan proses interaksi pada dashboard web.
  event.preventDefault();
  // Melanjutkan proses interaksi pada dashboard web.
  hideError();

  // Menyimpan data tetap atau elemen halaman ke `files`.
  const files = getSelectedFiles();
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (files.length !== 7) {
    // Melanjutkan proses interaksi pada dashboard web.
    showError("Input harus 7 file.");
    // Melanjutkan proses interaksi pada dashboard web.
    return;
  // Melanjutkan proses interaksi pada dashboard web.
  }

  // Menyimpan data tetap atau elemen halaman ke `formData`.
  const formData = new FormData();
  // Mengulang proses untuk setiap data yang ditampilkan di halaman.
  for (const file of files) {
    // Melanjutkan proses interaksi pada dashboard web.
    formData.append("files", file, file.name);
  // Melanjutkan proses interaksi pada dashboard web.
  }
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (thresholdInput.value !== "") {
    // Melanjutkan proses interaksi pada dashboard web.
    formData.append("threshold", thresholdInput.value);
  // Melanjutkan proses interaksi pada dashboard web.
  }
  // Melanjutkan proses interaksi pada dashboard web.
  formData.append("horizon", horizonInput.value);

  // Mengatur nilai `predictButton.disabled` untuk memperbarui data atau tampilan halaman.
  predictButton.disabled = true;
  // Mengubah tulisan yang terlihat pada dashboard.
  predictButton.textContent = "Memproses...";
  // Melanjutkan proses interaksi pada dashboard web.
  try {
    // Menyimpan data tetap atau elemen halaman ke `response`.
    const response = await fetch("/api/predict", {
      // Melanjutkan proses interaksi pada dashboard web.
      method: "POST",
      // Melanjutkan proses interaksi pada dashboard web.
      body: formData,
    // Melanjutkan proses interaksi pada dashboard web.
    });
    // Menyimpan data tetap atau elemen halaman ke `payload`.
    const payload = await response.json();
    // Mengecek syarat sebelum halaman menjalankan proses.
    if (!response.ok) {
      // Melanjutkan proses interaksi pada dashboard web.
      throw new Error(responseErrorText(payload));
    // Melanjutkan proses interaksi pada dashboard web.
    }
    // Melanjutkan proses interaksi pada dashboard web.
    renderResult(payload, files);
    // Melanjutkan proses interaksi pada dashboard web.
    await loadHistory();
  // Melanjutkan proses interaksi pada dashboard web.
  } catch (error) {
    // Melanjutkan proses interaksi pada dashboard web.
    showError(error.message || "Prediksi gagal.");
  // Melanjutkan proses interaksi pada dashboard web.
  } finally {
    // Mengubah tulisan yang terlihat pada dashboard.
    predictButton.textContent = "Prediksi";
    // Melanjutkan proses interaksi pada dashboard web.
    validateSelection();
  // Melanjutkan proses interaksi pada dashboard web.
  }
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `onHistoryClick` untuk interaksi halaman.
function onHistoryClick(event) {
  // Menyimpan data tetap atau elemen halaman ke `button`.
  const button = event.target.closest(".history-detail-button");
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!button) return;
  // Menyimpan data tetap atau elemen halaman ke `predictionId`.
  const predictionId = button.dataset.predictionId;
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!predictionId) return;
  // Melanjutkan proses interaksi pada dashboard web.
  loadPredictionDetail(predictionId);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `onSpotlightOpen` untuk interaksi halaman.
function onSpotlightOpen() {
  // Mengecek syarat sebelum halaman menjalankan proses.
  if (!spotlightPredictionId) return;
  // Melanjutkan proses interaksi pada dashboard web.
  loadPredictionDetail(spotlightPredictionId);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat langkah kerja `refreshDashboard` yang bisa menunggu jawaban dari backend.
async function refreshDashboard() {
  // Melanjutkan proses interaksi pada dashboard web.
  await Promise.all([loadRuntimeStatus(), loadHistory()]);
// Melanjutkan proses interaksi pada dashboard web.
}

// Membuat halaman merespons aksi pengguna, seperti klik atau upload.
filesInput.addEventListener("change", validateSelection);
// Membuat halaman merespons aksi pengguna, seperti klik atau upload.
form.addEventListener("submit", runPrediction);
// Membuat halaman merespons aksi pengguna, seperti klik atau upload.
resetButton.addEventListener("click", resetFormState);
// Membuat halaman merespons aksi pengguna, seperti klik atau upload.
refreshHistoryButton.addEventListener("click", loadHistory);
// Membuat halaman merespons aksi pengguna, seperti klik atau upload.
refreshRuntimeButton.addEventListener("click", loadRuntimeStatus);
// Membuat halaman merespons aksi pengguna, seperti klik atau upload.
historyBody.addEventListener("click", onHistoryClick);
// Membuat halaman merespons aksi pengguna, seperti klik atau upload.
spotlightOpenButton.addEventListener("click", onSpotlightOpen);

// Melanjutkan proses interaksi pada dashboard web.
resetResultSection();
// Melanjutkan proses interaksi pada dashboard web.
validateSelection();
// Melanjutkan proses interaksi pada dashboard web.
refreshDashboard();
