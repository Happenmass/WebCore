(() => {
const MODEL_URL = "https://static.webcore.xin/esrgan/RealESRGAN_x4.onnx";
const MODEL_CACHE_DB = "sr-model-cache-v1";
const MODEL_CACHE_STORE = "models";
const SCALE = 4;
const PATCH_SIZE = 168;
const PADDING = 24;
const PAD_SIZE = 15;
const BATCH_SIZE = 4;
const MAX_WASM_THREADS = 16;

const state = {
  session: null,
  sessionProvider: null,
  runBatchSize: BATCH_SIZE,
  image: null,
  originalRgba: null,
  outputRgba: null,
  outputSize: null,
  ortBuild: null,
  ortLoading: null,
  running: false,
};

const SR_DEBUG = true;
const SR_ORG = "sr";

const i18n = window.I18N;
const t = (key, fallback) => (i18n ? i18n.t(key, fallback) : fallback || key);

function debugLog(...args) {
  if (SR_DEBUG) {
    console.log("[SR]", ...args);
  }
}

function resetOrt(reason) {
  debugLog("reset ort", reason);
  window.ort = null;
  state.ortBuild = null;
  state.ortLoading = null;
  state.session = null;
  state.sessionProvider = null;
}

const els = {
  fileInput: document.getElementById("sr-file"),
  runBtn: document.getElementById("sr-run"),
  status: document.getElementById("sr-status"),
  compare: document.getElementById("sr-compare"),
  frame: document.getElementById("sr-frame"),
  origCanvas: document.getElementById("sr-orig-canvas"),
  outCanvas: document.getElementById("sr-out-canvas"),
  origClip: document.getElementById("sr-orig-clip"),
  divider: document.getElementById("sr-divider"),
  modelProgress: document.getElementById("sr-model-progress"),
  modelProgressBar: document.getElementById("sr-model-progress-bar"),
  progress: document.getElementById("sr-progress"),
  progressBar: document.getElementById("sr-progress-bar"),
  download: document.getElementById("sr-download"),
};

function setStatus(message, key) {
  if (els.status) {
    els.status.textContent = message;
    if (key) {
      els.status.dataset.statusKey = key;
    }
  }
}

function setModelProgress(value) {
  if (!els.modelProgress || !els.modelProgressBar) return;
  const clamped = Math.max(0, Math.min(1, value));
  els.modelProgress.style.visibility = clamped > 0 && clamped < 1 ? "visible" : "hidden";
  els.modelProgressBar.style.width = `${clamped * 100}%`;
}

function setProgress(value) {
  if (!els.progress || !els.progressBar) return;
  const clamped = Math.max(0, Math.min(1, value));
  els.progress.style.visibility = clamped > 0 && clamped < 1 ? "visible" : "hidden";
  els.progressBar.style.width = `${clamped * 100}%`;
}

function patchWebGPUAdapterInfo() {
  if (!navigator.gpu) return;
  if (typeof GPUAdapter !== "undefined" && !GPUAdapter.prototype.requestAdapterInfo) {
    GPUAdapter.prototype.requestAdapterInfo = function () {
      return Promise.resolve(
        this.info || { vendor: "", architecture: "", device: "", description: "" }
      );
    };
    return;
  }
  if (typeof navigator.gpu.requestAdapter === "function") {
    const original = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = async (...args) => {
      const adapter = await original(...args);
      if (adapter && !adapter.requestAdapterInfo) {
        adapter.requestAdapterInfo = () =>
          Promise.resolve(
            adapter.info || { vendor: "", architecture: "", device: "", description: "" }
          );
      }
      return adapter;
    };
  }
}

function loadOrtScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.async = true;
    script.onload = () => {
      debugLog("script loaded", url);
      resolve();
    };
    script.onerror = () => {
      debugLog("script load failed", url);
      reject(new Error(`Failed to load ${url}`));
    };
    document.head.appendChild(script);
  });
}

function removeOrtScripts() {
  const scripts = Array.from(
    document.querySelectorAll('script[src*="onnxruntime-web"]')
  );
  for (const script of scripts) {
    script.remove();
  }
  debugLog("removed ort scripts", scripts.map((s) => s.src));
}

async function ensureOrtLoaded(preferWebGPU) {
  const targetBuild = preferWebGPU ? "webgpu" : "wasm";
  debugLog("ensureOrtLoaded", { preferWebGPU, targetBuild, current: state.ortBuild });
  if (window.ort && !state.ortBuild) {
    state.ortBuild = window.ort?.env?.webgpu ? "webgpu" : "wasm";
    debugLog("detected ort build", state.ortBuild);
  }
  if (preferWebGPU && window.ort && state.ortBuild !== "webgpu") {
    debugLog("pre-existing ort found, resetting for webgpu");
    window.ort = null;
    state.ortBuild = null;
    removeOrtScripts();
  }
  if (window.ort && state.ortBuild === targetBuild) {
    return;
  }
  if (state.ortLoading) {
    await state.ortLoading;
    if (state.ortBuild === targetBuild) {
      return;
    }
  }
  const base = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  const webgpuUrl = `${base}ort.webgpu.min.js?v=${Date.now()}`;
  const wasmUrl = `${base}ort.min.js`;
  state.ortLoading = (async () => {
    if (window.ort && state.ortBuild && state.ortBuild !== targetBuild) {
      debugLog("resetting ort build", state.ortBuild);
      window.ort = null;
      removeOrtScripts();
    }
    if (preferWebGPU) {
      debugLog("loading webgpu build", webgpuUrl);
      await loadOrtScript(webgpuUrl);
      state.ortBuild = "webgpu";
    } else {
      debugLog("loading wasm build", wasmUrl);
      await loadOrtScript(wasmUrl);
      state.ortBuild = "wasm";
    }
  })();
  try {
    await state.ortLoading;
  } catch (err) {
    debugLog("ort loading error", err);
    throw err;
  }
  debugLog("ort loaded", {
    build: state.ortBuild,
    hasOrt: !!window.ort,
    version: window.ort?.version,
    hasWebgpuEnv: !!window.ort?.env?.webgpu,
    webgpuEnv: window.ort?.env?.webgpu,
    wasmEnv: window.ort?.env?.wasm,
  });
}

function reflectIndex(index, size) {
  if (index < 0) return -index - 1;
  if (index >= size) return 2 * size - 1 - index;
  return index;
}

function edgeIndex(index, size) {
  if (index < 0) return 0;
  if (index >= size) return size - 1;
  return index;
}

function padReflect(image, width, height, pad) {
  const newW = width + pad * 2;
  const newH = height + pad * 2;
  const out = new Uint8Array(newW * newH * 3);
  for (let y = 0; y < newH; y++) {
    const srcY = reflectIndex(y - pad, height);
    for (let x = 0; x < newW; x++) {
      const srcX = reflectIndex(x - pad, width);
      const srcIdx = (srcY * width + srcX) * 3;
      const dstIdx = (y * newW + x) * 3;
      out[dstIdx] = image[srcIdx];
      out[dstIdx + 1] = image[srcIdx + 1];
      out[dstIdx + 2] = image[srcIdx + 2];
    }
  }
  return { data: out, width: newW, height: newH };
}

function extendToMultiple(image, width, height, patchSize) {
  const xExtend = (patchSize - (height % patchSize)) % patchSize;
  const yExtend = (patchSize - (width % patchSize)) % patchSize;
  const newH = height + xExtend;
  const newW = width + yExtend;
  if (xExtend === 0 && yExtend === 0) {
    return { data: image, width, height };
  }
  const out = new Uint8Array(newW * newH * 3);
  for (let y = 0; y < newH; y++) {
    const srcY = Math.min(y, height - 1);
    for (let x = 0; x < newW; x++) {
      const srcX = Math.min(x, width - 1);
      const srcIdx = (srcY * width + srcX) * 3;
      const dstIdx = (y * newW + x) * 3;
      out[dstIdx] = image[srcIdx];
      out[dstIdx + 1] = image[srcIdx + 1];
      out[dstIdx + 2] = image[srcIdx + 2];
    }
  }
  return { data: out, width: newW, height: newH };
}

function padEdge(image, width, height, pad) {
  const newW = width + pad * 2;
  const newH = height + pad * 2;
  const out = new Uint8Array(newW * newH * 3);
  for (let y = 0; y < newH; y++) {
    const srcY = edgeIndex(y - pad, height);
    for (let x = 0; x < newW; x++) {
      const srcX = edgeIndex(x - pad, width);
      const srcIdx = (srcY * width + srcX) * 3;
      const dstIdx = (y * newW + x) * 3;
      out[dstIdx] = image[srcIdx];
      out[dstIdx + 1] = image[srcIdx + 1];
      out[dstIdx + 2] = image[srcIdx + 2];
    }
  }
  return { data: out, width: newW, height: newH };
}

function splitImageIntoOverlappingPatches(image, width, height, patchSize, padding) {
  const extended = extendToMultiple(image, width, height, patchSize);
  const padded = padEdge(extended.data, extended.width, extended.height, padding);

  const patches = [];
  const paddedH = padded.height;
  const paddedW = padded.width;
  const patchH = patchSize + padding * 2;
  const patchW = patchSize + padding * 2;

  for (let x = padding; x < paddedH - padding; x += patchSize) {
    for (let y = padding; y < paddedW - padding; y += patchSize) {
      const xLeft = x - padding;
      const yTop = y - padding;
      const patch = new Uint8Array(patchH * patchW * 3);
      for (let py = 0; py < patchH; py++) {
        for (let px = 0; px < patchW; px++) {
          const srcIdx = ((xLeft + py) * paddedW + (yTop + px)) * 3;
          const dstIdx = (py * patchW + px) * 3;
          patch[dstIdx] = padded.data[srcIdx];
          patch[dstIdx + 1] = padded.data[srcIdx + 1];
          patch[dstIdx + 2] = padded.data[srcIdx + 2];
        }
      }
      patches.push({ data: patch, width: patchW, height: patchH });
    }
  }

  return { patches, paddedShape: [paddedH, paddedW, 3] };
}

function nchwToHwc(input, height, width) {
  const out = new Float32Array(height * width * 3);
  const planeSize = height * width;
  for (let c = 0; c < 3; c++) {
    const offset = c * planeSize;
    for (let i = 0; i < planeSize; i++) {
      out[i * 3 + c] = input[offset + i];
    }
  }
  return out;
}

async function runOnnx(session, patches) {
  const outputs = [];
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const patchH = patches[0].height;
  const patchW = patches[0].width;

  const runBatchSize = state.runBatchSize || BATCH_SIZE;
  for (let i = 0; i < patches.length; i += runBatchSize) {
    const batch = patches.slice(i, i + runBatchSize);
    const actualBatch = batch.length;
    const inputData = new Float32Array(actualBatch * 3 * patchH * patchW);
    for (let b = 0; b < actualBatch; b++) {
      const patch = batch[b];
      const patchData = patch.data;
      const base = b * 3 * patchH * patchW;
      for (let y = 0; y < patchH; y++) {
        for (let x = 0; x < patchW; x++) {
          const srcIdx = (y * patchW + x) * 3;
          const dstIdx = base + y * patchW + x;
          inputData[dstIdx] = patchData[srcIdx] / 255.0;
          inputData[dstIdx + patchH * patchW] = patchData[srcIdx + 1] / 255.0;
          inputData[dstIdx + 2 * patchH * patchW] = patchData[srcIdx + 2] / 255.0;
        }
      }
    }

    const tensor = new ort.Tensor("float32", inputData, [
      actualBatch,
      3,
      patchH,
      patchW,
    ]);
    const result = await session.run({ [inputName]: tensor });
    const output = result[outputName];
    const outH = output.dims[2];
    const outW = output.dims[3];
    for (let b = 0; b < actualBatch; b++) {
      const offset = b * 3 * outH * outW;
      const slice = output.data.subarray(offset, offset + 3 * outH * outW);
      outputs.push({
        data: nchwToHwc(slice, outH, outW),
        width: outW,
        height: outH,
      });
    }
    setProgress(Math.min(1, (i + actualBatch) / patches.length));
  }

  return outputs;
}

function unpadPatch(patch, padding) {
  const newW = patch.width - padding * 2;
  const newH = patch.height - padding * 2;
  const out = new Float32Array(newW * newH * 3);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const srcIdx = ((y + padding) * patch.width + (x + padding)) * 3;
      const dstIdx = (y * newW + x) * 3;
      out[dstIdx] = patch.data[srcIdx];
      out[dstIdx + 1] = patch.data[srcIdx + 1];
      out[dstIdx + 2] = patch.data[srcIdx + 2];
    }
  }
  return { data: out, width: newW, height: newH };
}

function stitchTogether(patches, paddedShape, targetShape, padding) {
  const paddedH = paddedShape[0];
  const paddedW = paddedShape[1];
  const unpadded = patches.map((patch) => unpadPatch(patch, padding));
  const patchSize = unpadded[0].width;
  const patchesPerRow = Math.floor(paddedW / patchSize);
  const out = new Float32Array(paddedH * paddedW * 3);
  let row = -1;
  let col = 0;

  for (let i = 0; i < unpadded.length; i++) {
    if (i % patchesPerRow === 0) {
      row += 1;
      col = 0;
    }
    const patch = unpadded[i];
    for (let y = 0; y < patch.height; y++) {
      for (let x = 0; x < patch.width; x++) {
        const dstIdx = ((row * patchSize + y) * paddedW + (col * patchSize + x)) * 3;
        const srcIdx = (y * patch.width + x) * 3;
        out[dstIdx] = patch.data[srcIdx];
        out[dstIdx + 1] = patch.data[srcIdx + 1];
        out[dstIdx + 2] = patch.data[srcIdx + 2];
      }
    }
    col += 1;
  }

  const targetH = targetShape[0];
  const targetW = targetShape[1];
  const cropped = new Float32Array(targetH * targetW * 3);
  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const srcIdx = (y * paddedW + x) * 3;
      const dstIdx = (y * targetW + x) * 3;
      cropped[dstIdx] = out[srcIdx];
      cropped[dstIdx + 1] = out[srcIdx + 1];
      cropped[dstIdx + 2] = out[srcIdx + 2];
    }
  }

  return { data: cropped, width: targetW, height: targetH };
}

function unpadImage(image, width, height, pad) {
  const newW = width - pad * 2;
  const newH = height - pad * 2;
  const out = new Float32Array(newW * newH * 3);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const srcIdx = ((y + pad) * width + (x + pad)) * 3;
      const dstIdx = (y * newW + x) * 3;
      out[dstIdx] = image[srcIdx];
      out[dstIdx + 1] = image[srcIdx + 1];
      out[dstIdx + 2] = image[srcIdx + 2];
    }
  }
  return { data: out, width: newW, height: newH };
}

function floatToUint8(image) {
  const out = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0; i < image.width * image.height; i++) {
    const base = i * 3;
    const outIdx = i * 4;
    out[outIdx] = Math.min(255, Math.max(0, Math.round(image.data[base] * 255)));
    out[outIdx + 1] = Math.min(255, Math.max(0, Math.round(image.data[base + 1] * 255)));
    out[outIdx + 2] = Math.min(255, Math.max(0, Math.round(image.data[base + 2] * 255)));
    out[outIdx + 3] = 255;
  }
  return out;
}

const scratchCanvas = document.createElement("canvas");

function drawToCanvas(canvas, image, width, height, targetWidth, targetHeight) {
  if (!canvas) return;
  scratchCanvas.width = width;
  scratchCanvas.height = height;
  const scratchCtx = scratchCanvas.getContext("2d");
  scratchCtx.putImageData(new ImageData(image, width, height), 0, 0);

  const drawWidth = targetWidth || width;
  const drawHeight = targetHeight || height;
  canvas.width = drawWidth;
  canvas.height = drawHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, drawWidth, drawHeight);
  ctx.imageSmoothingEnabled = true;
  const scale = Math.min(drawWidth / width, drawHeight / height);
  const outW = width * scale;
  const outH = height * scale;
  const dx = (drawWidth - outW) / 2;
  const dy = (drawHeight - outH) / 2;
  ctx.drawImage(scratchCanvas, dx, dy, outW, outH);
}

function renderPreviews() {
  if (!els.frame) return;
  const frameWidth = els.frame.clientWidth;
  const frameHeight = els.frame.clientHeight;
  if (!frameWidth || !frameHeight) return;
  if (state.originalRgba && state.image) {
    drawToCanvas(
      els.origCanvas,
      state.originalRgba,
      state.image.width,
      state.image.height,
      frameWidth,
      frameHeight
    );
  }
  if (state.outputRgba && state.outputSize) {
    drawToCanvas(
      els.outCanvas,
      state.outputRgba,
      state.outputSize.width,
      state.outputSize.height,
      frameWidth,
      frameHeight
    );
  } else if (els.outCanvas) {
    els.outCanvas.width = frameWidth;
    els.outCanvas.height = frameHeight;
    const ctx = els.outCanvas.getContext("2d");
    ctx.clearRect(0, 0, frameWidth, frameHeight);
  }
}

function openModelCache() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(MODEL_CACHE_DB, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(MODEL_CACHE_STORE)) {
        db.createObjectStore(MODEL_CACHE_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readModelFromCache(key) {
  const db = await openModelCache();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(MODEL_CACHE_STORE, "readonly");
    const store = tx.objectStore(MODEL_CACHE_STORE);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
}

async function writeModelToCache(key, data) {
  const db = await openModelCache();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(MODEL_CACHE_STORE, "readwrite");
    const store = tx.objectStore(MODEL_CACHE_STORE);
    const request = store.put(data, key);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

async function fetchModelWithProgress(url) {
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error("Failed to download model");
  }
  const contentLength = Number(response.headers.get("Content-Length") || 0);
  const reader = response.body.getReader();
  let received = 0;
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (contentLength) {
      setModelProgress(received / contentLength);
    }
  }

  setModelProgress(1);
  return new Blob(chunks).arrayBuffer();
}

async function loadModelIfNeeded() {
  if (state.session) return;
  setStatus(
    t("sr_status_model_loading", "模型加载中..."),
    "sr_status_model_loading"
  );
  setModelProgress(0);
  debugLog("navigator.gpu", !!navigator.gpu, navigator.gpu);
  debugLog("secure context", window.isSecureContext, location.origin);
  debugLog("crossOriginIsolated", self.crossOriginIsolated);
  patchWebGPUAdapterInfo();
  await ensureOrtLoaded(true);
  if (navigator.gpu?.requestAdapter) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      debugLog("webgpu adapter", adapter ? "ok" : "null");
      if (adapter?.requestAdapterInfo) {
        const info = await adapter.requestAdapterInfo();
        debugLog("adapter info", info);
      }
    } catch (err) {
      debugLog("requestAdapter error", err);
    }
  }
  const maxThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;
  const wasmThreads = self.crossOriginIsolated
    ? Math.max(1, Math.min(maxThreads, MAX_WASM_THREADS))
    : 1;
  ort.env.wasm.numThreads = wasmThreads;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  ort.env.wasm.proxy = false;
  ort.env.webgpu = ort.env.webgpu || {};
  ort.env.webgpu.powerPreference = "high-performance";
  const providers =
    state.ortBuild === "webgpu" && window.ort?.env?.webgpu ? ["webgpu", "wasm"] : ["wasm"];
  debugLog("providers request", providers, "ortBuild", state.ortBuild);
  try {
    let modelData = await readModelFromCache(MODEL_URL);
    if (!modelData) {
      setStatus(
        t("sr_status_model_downloading", "模型下载中..."),
        "sr_status_model_downloading"
      );
      modelData = await fetchModelWithProgress(MODEL_URL);
      await writeModelToCache(MODEL_URL, modelData);
    }
    state.session = await ort.InferenceSession.create(modelData, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
    });
    state.sessionProvider = state.session.executionProviders?.[0] || state.ortBuild || "wasm";
    debugLog("session created", state.sessionProvider, state.session.executionProviders);
    state.runBatchSize = state.sessionProvider.startsWith("wasm") ? 1 : BATCH_SIZE;
    setStatus(t("sr_status_model_loaded", "模型加载成功"), "sr_status_model_loaded");
  } catch (err) {
    debugLog("webgpu session error", err);
    const message = String(err?.message || err);
    if (message.includes("backend not found") && state.ortBuild === "webgpu") {
      window.ort = null;
      state.ortBuild = null;
      state.ortLoading = null;
      await ensureOrtLoaded(false);
    }
    let modelData = await readModelFromCache(MODEL_URL);
    if (!modelData) {
      setStatus(
        t("sr_status_model_downloading", "模型下载中..."),
        "sr_status_model_downloading"
      );
      modelData = await fetchModelWithProgress(MODEL_URL);
      await writeModelToCache(MODEL_URL, modelData);
    }
    state.session = await ort.InferenceSession.create(modelData, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    state.sessionProvider = "wasm";
    debugLog("session created fallback", state.sessionProvider);
    state.runBatchSize = 1;
    setStatus(t("sr_status_model_loaded", "模型加载成功"), "sr_status_model_loaded");
  }
  setModelProgress(1);
}

async function runSuperResolution() {
  if (state.running) return;
  if (!state.image) {
    setStatus(t("sr_status_need_image", "请先选择图片"), "sr_status_need_image");
    return;
  }
  state.running = true;
  if (els.runBtn) {
    els.runBtn.disabled = true;
  }
  try {
    if (window.__ortOwner !== SR_ORG) {
      resetOrt("switch to sr");
      window.__ortOwner = SR_ORG;
    }
    await loadModelIfNeeded();
    setStatus(t("sr_status_running", "推理中..."), "sr_status_running");
    setProgress(0);
    const image = state.image;
    const padded = padReflect(image.data, image.width, image.height, PAD_SIZE);
    const { patches, paddedShape } = splitImageIntoOverlappingPatches(
      padded.data,
      padded.width,
      padded.height,
      PATCH_SIZE,
      PADDING
    );

    const outputs = await runOnnx(state.session, patches);
    const paddedScaled = [paddedShape[0] * SCALE, paddedShape[1] * SCALE, 3];
    const scaledShape = [padded.height * SCALE, padded.width * SCALE, 3];
    const stitched = stitchTogether(outputs, paddedScaled, scaledShape, PADDING * SCALE);
    const unpadded = unpadImage(stitched.data, stitched.width, stitched.height, PAD_SIZE * SCALE);
    const rgba = floatToUint8(unpadded);

    state.outputRgba = rgba;
    state.outputSize = { width: unpadded.width, height: unpadded.height };
    renderPreviews();
    if (els.download) {
      els.download.disabled = false;
    }
    setProgress(1);
    setStatus(t("sr_status_success", "推理成功"), "sr_status_success");
  } catch (err) {
    console.error(err);
    setProgress(0);
    setStatus(t("sr_status_failed", "推理失败"), "sr_status_failed");
  } finally {
    state.running = false;
    if (els.runBtn) {
      els.runBtn.disabled = false;
    }
  }
}

function readImageFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function handleImageUpload(file) {
  const img = await readImageFile(file);
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, img.width, img.height);
  const rgb = new Uint8Array(img.width * img.height * 3);
  for (let i = 0, j = 0; i < imageData.data.length; i += 4) {
    rgb[j++] = imageData.data[i];
    rgb[j++] = imageData.data[i + 1];
    rgb[j++] = imageData.data[i + 2];
  }
  state.image = { data: rgb, width: img.width, height: img.height };
  state.originalRgba = new Uint8ClampedArray(imageData.data);
  state.outputRgba = null;
  state.outputSize = null;
  renderPreviews();
  if (els.download) {
    els.download.disabled = true;
  }
  if (els.runBtn) {
    els.runBtn.disabled = false;
  }
}

function downloadResult() {
  if (!state.outputRgba || !state.outputSize) return;
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = state.outputSize.width;
  exportCanvas.height = state.outputSize.height;
  const exportCtx = exportCanvas.getContext("2d");
  exportCtx.putImageData(
    new ImageData(state.outputRgba, state.outputSize.width, state.outputSize.height),
    0,
    0
  );
  exportCanvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "sr_x4.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }, "image/png");
}

if (els.runBtn) {
  els.runBtn.disabled = true;
  els.runBtn.addEventListener("click", () => {
    runSuperResolution();
  });
}

setProgress(0);
setModelProgress(0);

if (els.fileInput) {
  els.fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    handleImageUpload(file).catch((err) => {
      console.error(err);
      setStatus(
        t("sr_status_image_failed", "图片读取失败"),
        "sr_status_image_failed"
      );
    });
  });
}

if (els.download) {
  els.download.addEventListener("click", () => {
    downloadResult();
  });
}

function setDivider(ratio) {
  if (!els.origClip || !els.divider) return;
  const clamped = Math.max(0, Math.min(1, ratio));
  const percent = `${clamped * 100}%`;
  els.origClip.style.clipPath = `inset(0 ${100 - clamped * 100}% 0 0)`;
  els.origClip.style.webkitClipPath = `inset(0 ${100 - clamped * 100}% 0 0)`;
  els.divider.style.left = percent;
}

function enableDrag() {
  if (!els.frame || !els.divider) return;
  let dragging = false;
  const onMove = (event) => {
    if (!dragging) return;
    const rect = els.frame.getBoundingClientRect();
    const clientX = event.touches ? event.touches[0].clientX : event.clientX;
    const x = clientX - rect.left;
    setDivider(x / rect.width);
  };
  const stopDrag = () => {
    dragging = false;
  };
  const startDrag = (event) => {
    dragging = true;
    onMove(event);
  };
  els.divider.addEventListener("pointerdown", startDrag);
  els.frame.addEventListener("pointerdown", startDrag);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", stopDrag);
  window.addEventListener("pointerleave", stopDrag);
  window.addEventListener("touchmove", onMove, { passive: false });
  window.addEventListener("touchend", stopDrag);
}

enableDrag();
setDivider(0.5);

window.addEventListener("resize", () => {
  renderPreviews();
});
})();
