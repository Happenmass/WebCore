const MODEL_URL = "../weights/RealESRGAN_x4.onnx";
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
  dividerRatio: 0.5,
  ortBuild: null,
  ortLoading: null,
};

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

const els = {
  modelStatus: document.getElementById("modelStatus"),
  imageInput: document.getElementById("imageInput"),
  loadModel: document.getElementById("loadModel"),
  runSr: document.getElementById("runSr"),
  log: document.getElementById("log"),
  srCanvas: document.getElementById("srCanvas"),
  origCanvas: document.getElementById("origCanvas"),
  origClip: document.getElementById("origClip"),
  divider: document.getElementById("divider"),
  compareFrame: document.getElementById("compareFrame"),
};

function log(message) {
  els.log.textContent = `${message}\n${els.log.textContent}`;
}

function setStatus(message) {
  els.modelStatus.textContent = message;
}

function loadOrtScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${url}`));
    document.head.appendChild(script);
  });
}

async function ensureOrtLoaded(preferWebGPU) {
  if (window.ort && state.ortBuild) {
    return;
  }
  if (state.ortLoading) {
    await state.ortLoading;
    return;
  }
  const base = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  const webgpuUrl = `${base}ort.webgpu.min.js`;
  const wasmUrl = `${base}ort.min.js`;
  state.ortLoading = (async () => {
    if (preferWebGPU) {
      await loadOrtScript(webgpuUrl);
      state.ortBuild = "webgpu";
    } else {
      await loadOrtScript(wasmUrl);
      state.ortBuild = "wasm";
    }
  })();
  await state.ortLoading;
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

function drawToCanvas(canvas, image, width, height) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(image, width, height);
  ctx.putImageData(imageData, 0, 0);
}

function drawOriginalScaled(width, height) {
  if (!state.originalRgba) return;
  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = state.image.width;
  srcCanvas.height = state.image.height;
  const srcCtx = srcCanvas.getContext("2d");
  srcCtx.putImageData(
    new ImageData(state.originalRgba, state.image.width, state.image.height),
    0,
    0
  );

  els.origCanvas.width = width;
  els.origCanvas.height = height;
  const dstCtx = els.origCanvas.getContext("2d");
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.drawImage(srcCanvas, 0, 0, width, height);
}

function setCompareSize(width, height) {
  els.compareFrame.style.aspectRatio = `${width} / ${height}`;
}

function setDivider(ratio) {
  state.dividerRatio = Math.max(0, Math.min(1, ratio));
  const percent = `${state.dividerRatio * 100}%`;
  const clipValue = `inset(0 ${100 - state.dividerRatio * 100}% 0 0)`;
  els.origClip.style.clipPath = clipValue;
  els.origClip.style.webkitClipPath = clipValue;
  els.divider.style.left = percent;
}

async function loadModel() {
  if (state.session) {
    return;
  }
  setStatus("Loading model...");
  patchWebGPUAdapterInfo();
  await ensureOrtLoaded(true);
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
  const providers = ["webgpu", "wasm"];
  try {
    state.session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
    });
    state.sessionProvider = state.session.executionProviders?.[0] || "webgpu";
    state.runBatchSize = state.sessionProvider.startsWith("wasm") ? 1 : BATCH_SIZE;
    setStatus(`Model loaded (${state.sessionProvider})`);
  } catch (err) {
    const message = String(err?.message || err);
    if (message.includes("backend not found") && state.ortBuild === "webgpu") {
      window.ort = null;
      state.ortBuild = null;
      state.ortLoading = null;
      await ensureOrtLoaded(false);
    }
    state.session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    state.sessionProvider = "wasm";
    state.runBatchSize = 1;
    setStatus("Model loaded (wasm)");
  }
  els.runSr.disabled = !state.image;
}

async function runSuperResolution() {
  if (!state.session || !state.image) return;
  els.runSr.disabled = true;
  setStatus("Running ESRGAN...");
  log("Preparing patches...");

  const image = state.image;
  const padded = padReflect(image.data, image.width, image.height, PAD_SIZE);
  const { patches, paddedShape } = splitImageIntoOverlappingPatches(
    padded.data,
    padded.width,
    padded.height,
    PATCH_SIZE,
    PADDING
  );

  log(`Patches: ${patches.length}`);
  const outputs = await runOnnx(state.session, patches);
  log("Stitching patches...");

  const paddedScaled = [paddedShape[0] * SCALE, paddedShape[1] * SCALE, 3];
  const scaledShape = [padded.height * SCALE, padded.width * SCALE, 3];
  const stitched = stitchTogether(outputs, paddedScaled, scaledShape, PADDING * SCALE);
  const unpadded = unpadImage(stitched.data, stitched.width, stitched.height, PAD_SIZE * SCALE);
  const rgba = floatToUint8(unpadded);

  drawToCanvas(els.srCanvas, rgba, unpadded.width, unpadded.height);
  drawOriginalScaled(unpadded.width, unpadded.height);
  setCompareSize(unpadded.width, unpadded.height);
  setDivider(0.5);
  setStatus("Done");
  els.runSr.disabled = false;
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
  drawToCanvas(els.origCanvas, state.originalRgba, img.width, img.height);
  drawToCanvas(els.srCanvas, state.originalRgba, img.width, img.height);
  setCompareSize(img.width, img.height);
  setDivider(0.5);
  els.runSr.disabled = !state.session;
}

function enableDrag() {
  let dragging = false;
  const onMove = (event) => {
    if (!dragging) return;
    const rect = els.compareFrame.getBoundingClientRect();
    const x = (event.touches ? event.touches[0].clientX : event.clientX) - rect.left;
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
  els.compareFrame.addEventListener("pointerdown", startDrag);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", stopDrag);
  window.addEventListener("pointerleave", stopDrag);
  window.addEventListener("touchmove", onMove);
  window.addEventListener("touchend", stopDrag);
}

els.loadModel.addEventListener("click", () => {
  loadModel().catch((err) => {
    console.error(err);
    setStatus("Failed to load model");
    log(`Model load error: ${err.message}`);
  });
});

els.runSr.addEventListener("click", () => {
  runSuperResolution().catch((err) => {
    console.error(err);
    setStatus("Failed");
    log(`Inference error: ${err.message}`);
    els.runSr.disabled = false;
  });
});

els.imageInput.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  handleImageUpload(file).catch((err) => {
    console.error(err);
    log(`Image load error: ${err.message}`);
  });
});

enableDrag();
setDivider(0.5);
