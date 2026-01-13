(() => {
const MODEL_URL = "https://static.webcore.xin/paraformer/model_quant.onnx";
const CONFIG_URL = "https://static.webcore.xin/paraformer/config.yaml";
const TOKENS_URL = "https://static.webcore.xin/paraformer/tokens.json";
const CMVN_URL = "https://static.webcore.xin/paraformer/am.mvn";
const PUNC_MODEL_URL = "https://static.webcore.xin/punc/model.int8.onnx";
const PUNC_TOKENS_URL = "https://static.webcore.xin/punc/tokens.txt";
const PUNC_CONF_URL = "https://static.webcore.xin/punc/punc.json";
const MODEL_CACHE_DB = "asr-model-cache-v1";
const MODEL_CACHE_STORE = "models";

const DEFAULT_CONFIG = {
  sampleRate: 16000,
  nMels: 80,
  frameLengthMs: 25,
  frameShiftMs: 10,
  lfrM: 7,
  lfrN: 6,
  predictorBias: 1,
  maxChunkSeconds: 30,
};

const ASR_ORG = "asr";

const state = {
  session: null,
  puncSession: null,
  tokens: null,
  cmvn: null,
  config: { ...DEFAULT_CONFIG },
  audioData: null,
  audioInfo: null,
  ortBuild: null,
  ortLoading: null,
  modelLoading: null,
  pendingRun: false,
  recording: false,
  mediaRecorder: null,
  recordChunks: [],
  recordStream: null,
};

const i18n = window.I18N;
const t = (key, fallback) => (i18n ? i18n.t(key, fallback) : fallback || key);

const els = {
  fileInput: document.getElementById("asr-file"),
  recordBtn: document.getElementById("asr-record"),
  stopBtn: document.getElementById("asr-stop"),
  runBtn: document.getElementById("asr-run"),
  status: document.getElementById("asr-status"),
  result: document.getElementById("asr-result"),
  copy: document.getElementById("asr-copy"),
  download: document.getElementById("asr-download"),
  modelProgress: document.getElementById("asr-model-progress"),
  modelProgressBar: document.getElementById("asr-model-progress-bar"),
};

function setStatus(message, key) {
  if (els.status) {
    els.status.textContent = message;
    if (key) {
      els.status.dataset.statusKey = key;
    }
  }
}

function setRecordingUI(active) {
  if (els.stopBtn) {
    els.stopBtn.classList.toggle("is-recording", active);
  }
}

function setModelProgress(value) {
  if (!els.modelProgress || !els.modelProgressBar) return;
  const clamped = Math.max(0, Math.min(1, value));
  els.modelProgress.style.visibility =
    clamped > 0 && clamped < 1 ? "visible" : "hidden";
  els.modelProgressBar.style.width = `${clamped * 100}%`;
}

function setResult(text) {
  if (!els.result) return;
  if (!text) {
    els.result.textContent = t("asr_result_placeholder", "识别文本将在这里显示…");
    els.result.dataset.empty = "true";
    return;
  }
  els.result.textContent = text;
  els.result.dataset.empty = "false";
}

function setResultActions(enabled) {
  if (els.copy) els.copy.disabled = !enabled;
  if (els.download) els.download.disabled = !enabled;
}

function resetOrt(reason) {
  console.log("[ASR] reset ort", reason);
  window.ort = null;
  state.ortBuild = null;
  state.ortLoading = null;
  state.session = null;
  state.puncSession = null;
  const scripts = Array.from(
    document.querySelectorAll('script[src*="onnxruntime-web"]')
  );
  for (const script of scripts) {
    script.remove();
  }
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

function openModelCache() {
  if (!window.indexedDB) return Promise.resolve(null);
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
  if (!db) return null;
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
  if (!db) return;
  return new Promise((resolve, reject) => {
    const tx = db.transaction(MODEL_CACHE_STORE, "readwrite");
    const store = tx.objectStore(MODEL_CACHE_STORE);
    const request = store.put(data, key);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

async function fetchModelWithProgress(url, onProgress) {
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
    if (contentLength && onProgress) {
      onProgress(received / contentLength);
    }
  }

  if (onProgress) onProgress(1);
  return new Blob(chunks).arrayBuffer();
}

async function ensureOrtLoaded() {
  if (window.ort && state.ortBuild === "wasm") return;
  if (state.ortLoading) {
    await state.ortLoading;
    return;
  }
  const url = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js";
  state.ortLoading = (async () => {
    await loadOrtScript(url);
    state.ortBuild = "wasm";
  })();
  await state.ortLoading;
}

function parseConfig(text) {
  const nextNumber = (key, fallback) => {
    const match = text.match(new RegExp(`${key}:\\s*([0-9.]+)`));
    return match ? Number(match[1]) : fallback;
  };
  const sampleRate = nextNumber("fs", DEFAULT_CONFIG.sampleRate);
  return {
    sampleRate,
    nMels: nextNumber("n_mels", DEFAULT_CONFIG.nMels),
    frameLengthMs: nextNumber("frame_length", DEFAULT_CONFIG.frameLengthMs),
    frameShiftMs: nextNumber("frame_shift", DEFAULT_CONFIG.frameShiftMs),
    lfrM: nextNumber("lfr_m", DEFAULT_CONFIG.lfrM),
    lfrN: nextNumber("lfr_n", DEFAULT_CONFIG.lfrN),
    predictorBias: nextNumber("predictor_bias", DEFAULT_CONFIG.predictorBias),
    maxChunkSeconds: DEFAULT_CONFIG.maxChunkSeconds,
  };
}

function extractBracketArray(text, tag) {
  const tagIndex = text.indexOf(tag);
  if (tagIndex === -1) return null;
  const start = text.indexOf("[", tagIndex);
  const end = text.indexOf("]", start + 1);
  if (start === -1 || end === -1) return null;
  const content = text.slice(start + 1, end).trim();
  if (!content) return null;
  return content.split(/\s+/).map(Number);
}

async function loadResources() {
  const [configText, tokens, cmvnText] = await Promise.all([
    fetch(CONFIG_URL).then((res) => res.text()),
    fetch(TOKENS_URL).then((res) => res.json()),
    fetch(CMVN_URL).then((res) => res.text()),
  ]);

  const config = parseConfig(configText);
  const addShift = extractBracketArray(cmvnText, "<AddShift>");
  const rescale = extractBracketArray(cmvnText, "<Rescale>");

  if (!addShift || !rescale) {
    throw new Error("Failed to parse CMVN file.");
  }
  if (addShift.length !== rescale.length) {
    throw new Error("CMVN mean/scale length mismatch.");
  }

  state.tokens = tokens;
  state.cmvn = { mean: addShift, scale: rescale };
  state.config = { ...DEFAULT_CONFIG, ...config };
}

function buildVocab(tokenLines) {
  const vocab = new Map();
  tokenLines.forEach((token, index) => {
    if (!token) return;
    vocab.set(token, index);
  });
  return vocab;
}

async function loadPuncResources() {
  const [tokensText, punc] = await Promise.all([
    fetch(PUNC_TOKENS_URL).then((res) => res.text()),
    fetch(PUNC_CONF_URL).then((res) => res.json()),
  ]);
  const lines = tokensText
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  state.puncVocab = buildVocab(lines);
  state.puncList = punc.punc_list;
}

async function loadModel() {
  if (state.session) return;
  if (state.modelLoading) {
    await state.modelLoading;
    return;
  }
  setStatus(
    t("asr_status_model_loading", "模型加载中..."),
    "asr_status_model_loading"
  );
  setModelProgress(0);
  state.modelLoading = (async () => {
    await ensureOrtLoaded();
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8))
      : 1;
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
    ort.env.wasm.proxy = false;
    let modelData = await readModelFromCache(MODEL_URL);
    if (!modelData) {
      modelData = await fetchModelWithProgress(MODEL_URL, (progress) => {
        setModelProgress(progress * 0.75);
      });
      await writeModelToCache(MODEL_URL, modelData);
    }
    state.session = await ort.InferenceSession.create(modelData, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    state.sessionProvider = "wasm";

    await Promise.all([
      loadResources(),
      loadPuncResources(),
      (async () => {
        let puncData = await readModelFromCache(PUNC_MODEL_URL);
        if (!puncData) {
          puncData = await fetchModelWithProgress(PUNC_MODEL_URL, (progress) => {
            setModelProgress(0.75 + progress * 0.25);
          });
          await writeModelToCache(PUNC_MODEL_URL, puncData);
        }
        state.puncSession = await ort.InferenceSession.create(puncData, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      })(),
    ]);
    setModelProgress(1);
    setStatus(t("asr_status_model_loaded", "模型已加载"), "asr_status_model_loaded");
    if (els.runBtn) {
      els.runBtn.disabled = !state.audioData;
    }
  })();
  try {
    await state.modelLoading;
  } finally {
    state.modelLoading = null;
  }
}

function mixToMono(audioBuffer) {
  const channels = audioBuffer.numberOfChannels;
  if (channels === 1) {
    return audioBuffer.getChannelData(0);
  }
  const length = audioBuffer.length;
  const out = new Float32Array(length);
  for (let c = 0; c < channels; c++) {
    const data = audioBuffer.getChannelData(c);
    for (let i = 0; i < length; i++) {
      out[i] += data[i];
    }
  }
  for (let i = 0; i < length; i++) {
    out[i] /= channels;
  }
  return out;
}

async function decodeAudioData(arrayBuffer) {
  const audioCtx = new AudioContext();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const inputRate = audioBuffer.sampleRate;
  let mono = mixToMono(audioBuffer);

  if (inputRate !== state.config.sampleRate) {
    const targetRate = state.config.sampleRate;
    const frameCount = Math.ceil(audioBuffer.duration * targetRate);
    const offline = new OfflineAudioContext(1, frameCount, targetRate);
    const source = offline.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();
    mono = rendered.getChannelData(0);
  }

  audioCtx.close();
  return {
    samples: mono,
    sampleRate: state.config.sampleRate,
    duration: mono.length / state.config.sampleRate,
  };
}

async function handleAudioBuffer(arrayBuffer, label) {
  setStatus(
    t("asr_status_audio_parsing", "音频解析中..."),
    "asr_status_audio_parsing"
  );
  try {
    const decoded = await decodeAudioData(arrayBuffer);
    state.audioData = decoded.samples;
    state.audioInfo = decoded;
    if (state.session) {
      setStatus(t("asr_status_ready", "准备就绪"), "asr_status_ready");
    } else {
      setStatus(t("asr_status_audio_loaded", "音频已加载"), "asr_status_audio_loaded");
    }
    if (els.runBtn) {
      els.runBtn.disabled = false;
    }
    setResult("");
    setResultActions(false);
    console.log("[ASR] audio loaded", label, decoded.duration);
  } catch (err) {
    console.error(err);
    setStatus(t("asr_status_audio_failed", "音频解析失败"), "asr_status_audio_failed");
  }
}

function nextPow2(value) {
  let pow = 1;
  while (pow < value) {
    pow <<= 1;
  }
  return pow;
}

function hammingWindow(length) {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (length - 1));
  }
  return window;
}

function fftRadix2(re, im) {
  const n = re.length;
  const levels = Math.log2(n);
  if (Math.floor(levels) !== levels) {
    throw new Error("FFT length must be power of two.");
  }

  for (let i = 0; i < n; i++) {
    let j = 0;
    for (let bit = 0; bit < levels; bit++) {
      j = (j << 1) | ((i >>> bit) & 1);
    }
    if (j > i) {
      const tempRe = re[i];
      const tempIm = im[i];
      re[i] = re[j];
      im[i] = im[j];
      re[j] = tempRe;
      im[j] = tempIm;
    }
  }

  for (let size = 2; size <= n; size <<= 1) {
    const halfSize = size >>> 1;
    const tableStep = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const k = j * tableStep;
        const angle = (-2 * Math.PI * k) / n;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const tRe = re[i + j + halfSize] * cos - im[i + j + halfSize] * sin;
        const tIm = re[i + j + halfSize] * sin + im[i + j + halfSize] * cos;
        re[i + j + halfSize] = re[i + j] - tRe;
        im[i + j + halfSize] = im[i + j] - tIm;
        re[i + j] += tRe;
        im[i + j] += tIm;
      }
    }
  }
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelFilterbank(nMels, nFft, sampleRate) {
  const fMin = 0;
  const fMax = sampleRate / 2;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = [];
  for (let i = 0; i < nMels + 2; i++) {
    melPoints.push(melMin + ((melMax - melMin) * i) / (nMels + 1));
  }
  const hzPoints = melPoints.map(melToHz);
  const bins = hzPoints.map((hz) => Math.floor(((nFft + 1) * hz) / sampleRate));
  const fftBins = nFft / 2 + 1;
  const filters = new Array(nMels);
  for (let i = 0; i < nMels; i++) {
    const filter = new Float32Array(fftBins);
    const left = bins[i];
    const center = bins[i + 1];
    const right = bins[i + 2];
    for (let j = left; j < center; j++) {
      if (j >= 0 && j < fftBins) {
        filter[j] = (j - left) / (center - left);
      }
    }
    for (let j = center; j < right; j++) {
      if (j >= 0 && j < fftBins) {
        filter[j] = (right - j) / (right - center);
      }
    }
    filters[i] = filter;
  }
  return filters;
}

function computeFbank(samples, config) {
  const frameLength = Math.round((config.frameLengthMs / 1000) * config.sampleRate);
  const frameShift = Math.round((config.frameShiftMs / 1000) * config.sampleRate);
  const nFft = nextPow2(frameLength);
  const window = hammingWindow(frameLength);
  const filters = buildMelFilterbank(config.nMels, nFft, config.sampleRate);
  const fftBins = nFft / 2 + 1;

  const numFrames = Math.max(1, Math.floor((samples.length - frameLength) / frameShift) + 1);
  const feats = new Array(numFrames);

  for (let i = 0; i < numFrames; i++) {
    const start = i * frameShift;
    const frame = new Float32Array(nFft);
    for (let j = 0; j < frameLength; j++) {
      const sample = start + j < samples.length ? samples[start + j] : 0;
      frame[j] = sample * window[j];
    }

    const re = new Float32Array(nFft);
    const im = new Float32Array(nFft);
    re.set(frame);
    fftRadix2(re, im);

    const power = new Float32Array(fftBins);
    for (let j = 0; j < fftBins; j++) {
      power[j] = re[j] * re[j] + im[j] * im[j];
    }

    const mel = new Float32Array(config.nMels);
    for (let m = 0; m < config.nMels; m++) {
      const filter = filters[m];
      let sum = 0;
      for (let k = 0; k < fftBins; k++) {
        sum += filter[k] * power[k];
      }
      mel[m] = Math.log(Math.max(sum, 1e-10));
    }
    feats[i] = mel;
  }
  return feats;
}

function applyLfr(feats, m, n) {
  const totalFrames = feats.length;
  const lfrFrames = Math.ceil(totalFrames / n);
  const lfr = new Array(lfrFrames);

  for (let i = 0; i < lfrFrames; i++) {
    const start = i * n;
    const out = new Float32Array(m * feats[0].length);
    for (let j = 0; j < m; j++) {
      const idx = Math.min(start + j, totalFrames - 1);
      out.set(feats[idx], j * feats[0].length);
    }
    lfr[i] = out;
  }
  return lfr;
}

function applyCmvn(feats, cmvn) {
  const mean = cmvn.mean;
  const scale = cmvn.scale;
  const dim = mean.length;
  for (const frame of feats) {
    for (let i = 0; i < dim; i++) {
      frame[i] = (frame[i] + mean[i]) * scale[i];
    }
  }
}

function prepareInput(samples, config, cmvn) {
  const feats = computeFbank(samples, config);
  const lfr = applyLfr(feats, config.lfrM, config.lfrN);
  applyCmvn(lfr, cmvn);

  const featLen = lfr.length;
  const dim = lfr[0].length;
  const data = new Float32Array(featLen * dim);
  for (let i = 0; i < featLen; i++) {
    data.set(lfr[i], i * dim);
  }

  return {
    speech: new ort.Tensor("float32", data, [1, featLen, dim]),
    speechLengths: new ort.Tensor("int32", new Int32Array([featLen]), [1]),
  };
}

function decodeTokens(logitsTensor, tokenNumTensor, tokens, predBias) {
  const dims = logitsTensor.dims;
  const timeSteps = dims[1];
  const vocab = dims[2];
  const data = logitsTensor.data;
  const tokenNums = tokenNumTensor.data;
  const rawTokenNum = ArrayBuffer.isView(tokenNums) ? tokenNums[0] : tokenNums;
  const validTokenNum = Number(rawTokenNum) - predBias;
  if (!Number.isFinite(validTokenNum)) {
    throw new Error(`Invalid token_num: ${String(rawTokenNum)}`);
  }

  const ids = new Array(timeSteps);
  for (let t = 0; t < timeSteps; t++) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    const offset = t * vocab;
    for (let v = 0; v < vocab; v++) {
      const val = data[offset + v];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = v;
      }
    }
    ids[t] = maxIdx;
  }

  const cleaned = [];
  for (const id of ids) {
    if (id === 0 || id === 2) continue;
    cleaned.push(id);
  }

  const maxTokens = Math.min(cleaned.length, Math.max(0, validTokenNum));
  const finalIds = cleaned.slice(0, maxTokens);
  const textTokens = finalIds.map((id) => tokens[id] || "");
  let text = "";
  for (const token of textTokens) {
    if (token.endsWith("@@")) {
      text += token.slice(0, -2);
    } else {
      text += token;
    }
  }
  return text.trim();
}

function normalizePuncInput(text) {
  return text.replace(/[，。？！、；：]/g, " ").trim();
}

function tokenizePunc(text) {
  const tokens = [];
  const regex = /[A-Za-z0-9]+|[\u4e00-\u9fff]/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    tokens.push(match[0]);
  }
  return tokens;
}

function mapTokensToIds(tokens, vocab) {
  const ids = new Array(tokens.length);
  for (let i = 0; i < tokens.length; i++) {
    const token = /[A-Za-z0-9]+/.test(tokens[i])
      ? tokens[i].toLowerCase()
      : tokens[i];
    ids[i] = vocab.get(token) ?? 0;
  }
  return ids;
}

function joinTokensWithPunc(tokens, puncIds, puncList) {
  let output = "";
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const isEnglish = /[A-Za-z0-9]+/.test(token);
    const needsSpace = isEnglish && output.length > 0 && /[A-Za-z0-9]$/.test(output);
    if (needsSpace) {
      output += " ";
    }
    output += token;
    const punc = puncList[puncIds[i]] || "";
    if (punc && punc !== "_" && punc !== "<unk>") {
      output += punc;
    }
  }
  return output;
}

function argmaxPunc(logits, length, puncSize) {
  const results = new Array(length);
  for (let t = 0; t < length; t++) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    const offset = t * puncSize;
    for (let k = 0; k < puncSize; k++) {
      const value = logits[offset + k];
      if (value > maxVal) {
        maxVal = value;
        maxIdx = k;
      }
    }
    results[t] = maxIdx;
  }
  return results;
}

async function restorePunctuation(text) {
  if (!state.puncSession || !state.puncVocab || !state.puncList) {
    return text;
  }
  const cleaned = normalizePuncInput(text);
  if (!cleaned) return text;
  const tokens = tokenizePunc(cleaned);
  if (!tokens.length) return text;
  const ids = mapTokensToIds(tokens, state.puncVocab);
  const inputTensor = new ort.Tensor("int32", new Int32Array(ids), [1, ids.length]);
  const lenTensor = new ort.Tensor("int32", new Int32Array([ids.length]), [1]);
  const outputs = await state.puncSession.run({
    inputs: inputTensor,
    text_lengths: lenTensor,
  });
  const logits = outputs.logits;
  const length = logits.dims[1];
  const puncSize = logits.dims[2];
  const puncIds = argmaxPunc(logits.data, length, puncSize);
  return joinTokensWithPunc(tokens, puncIds, state.puncList);
}

async function runAsr() {
  if (!state.audioData) {
    setStatus(t("asr_status_need_audio", "请先上传音频或录音"), "asr_status_need_audio");
    return;
  }
  if (els.runBtn) {
    els.runBtn.disabled = true;
  }
  setStatus(t("asr_status_recognizing", "识别中..."), "asr_status_recognizing");
  try {
    if (window.__ortOwner !== ASR_ORG) {
      resetOrt("switch to asr");
      window.__ortOwner = ASR_ORG;
    }
    if (!state.session) {
      state.pendingRun = true;
      await loadModel();
      if (state.pendingRun && state.session) {
        state.pendingRun = false;
        return runAsr();
      }
      state.pendingRun = false;
    }
    if (!state.session) {
      setStatus(t("asr_status_model_failed", "模型加载失败"), "asr_status_model_failed");
      return;
    }
    const chunkSamples = Math.floor(
      state.config.sampleRate * state.config.maxChunkSeconds
    );
    const totalSamples = state.audioData.length;
    const chunks = Math.ceil(totalSamples / chunkSamples);

    let combined = "";
    for (let i = 0; i < chunks; i++) {
      const start = i * chunkSamples;
      const end = Math.min(totalSamples, start + chunkSamples);
      const slice = state.audioData.subarray(start, end);
      const { speech, speechLengths } = prepareInput(
        slice,
        state.config,
        state.cmvn
      );
      const outputs = await state.session.run({
        speech,
        speech_lengths: speechLengths,
      });

      const outputNames = state.session.outputNames?.length
        ? state.session.outputNames
        : Object.keys(outputs);
      const logits = outputs[outputNames[0]] || outputs.logits;
      const tokenNum = outputs[outputNames[1]] || outputs.token_num;
      if (!logits || !tokenNum) {
        throw new Error(`Missing outputs. Keys: ${Object.keys(outputs).join(", ")}`);
      }

      const text = decodeTokens(
        logits,
        tokenNum,
        state.tokens,
        state.config.predictorBias
      );
      const punctuated = text ? await restorePunctuation(text) : "";
      if (punctuated) {
        combined = combined ? `${combined} ${punctuated}` : punctuated;
      }
    }

    setResult(combined);
    setResultActions(Boolean(combined));
    setStatus(t("asr_status_done", "识别完成"), "asr_status_done");
  } catch (err) {
    console.error(err);
    setStatus(t("asr_status_failed", "识别失败"), "asr_status_failed");
  } finally {
    if (els.runBtn) {
      els.runBtn.disabled = false;
    }
  }
}

function pickMimeType() {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
  ];
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "";
}

async function startRecording() {
  if (state.recording) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus(t("asr_status_no_record", "浏览器不支持录音"), "asr_status_no_record");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.recordStream = stream;
    state.recordChunks = [];
    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    state.mediaRecorder = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size) {
        state.recordChunks.push(event.data);
      }
    };
    recorder.onstop = async () => {
      state.recording = false;
      if (els.recordBtn) els.recordBtn.disabled = false;
      if (els.stopBtn) els.stopBtn.disabled = true;
      setRecordingUI(false);
      const blob = new Blob(state.recordChunks, { type: recorder.mimeType });
      const arrayBuffer = await blob.arrayBuffer();
      await handleAudioBuffer(arrayBuffer, "recording");
      if (state.recordStream) {
        state.recordStream.getTracks().forEach((track) => track.stop());
        state.recordStream = null;
      }
      state.mediaRecorder = null;
    };
    recorder.start();
    state.recording = true;
    setStatus(t("asr_status_recording", "录音中..."), "asr_status_recording");
    if (els.recordBtn) els.recordBtn.disabled = true;
    if (els.stopBtn) els.stopBtn.disabled = false;
    setRecordingUI(true);
  } catch (err) {
    console.error(err);
    setStatus(t("asr_status_record_failed", "录音失败"), "asr_status_record_failed");
  }
}

function stopRecording() {
  if (!state.recording || !state.mediaRecorder) return;
  state.mediaRecorder.stop();
  setStatus(
    t("asr_status_record_stopped", "录音已停止，解析中..."),
    "asr_status_record_stopped"
  );
}

function copyResult() {
  if (!els.result) return;
  const text = els.result.textContent.trim();
  if (!text || els.result.dataset.empty === "true") return;
  navigator.clipboard.writeText(text).then(
    () => setStatus(t("asr_status_copied", "已复制"), "asr_status_copied"),
    () => setStatus(t("asr_status_copy_failed", "复制失败"), "asr_status_copy_failed")
  );
}

function downloadResult() {
  if (!els.result) return;
  const text = els.result.textContent.trim();
  if (!text || els.result.dataset.empty === "true") return;
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "asr.txt";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

if (els.fileInput) {
  els.fileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const arrayBuffer = await file.arrayBuffer();
    await handleAudioBuffer(arrayBuffer, file.name);
  });
}

if (els.recordBtn) {
  els.recordBtn.addEventListener("click", startRecording);
}

if (els.stopBtn) {
  els.stopBtn.addEventListener("click", stopRecording);
}

if (els.runBtn) {
  els.runBtn.addEventListener("click", runAsr);
}

if (els.copy) {
  els.copy.addEventListener("click", copyResult);
}

if (els.download) {
  els.download.addEventListener("click", downloadResult);
}

setStatus(t("status_model_unloaded", "模型未加载"), "status_model_unloaded");
setResult("");
setResultActions(false);
setModelProgress(0);
})();
