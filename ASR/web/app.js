const MODEL_URL =
  "../iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx/model_quant.onnx";
const CONFIG_URL =
  "../iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx/config.yaml";
const TOKENS_URL =
  "../iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx/tokens.json";
const CMVN_URL =
  "../iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx/am.mvn";
const PUNC_MODEL_URL =
  "../manyeyes/alicttransformerpunc-zh-en-mge-int8-onnx/model.int8.onnx";
const PUNC_TOKENS_URL =
  "../manyeyes/alicttransformerpunc-zh-en-mge-int8-onnx/tokens.txt";
const PUNC_CONF_URL =
  "../manyeyes/alicttransformerpunc-zh-en-mge-int8-onnx/punc.json";

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

const state = {
  session: null,
  ortBuild: null,
  ortLoading: null,
  puncSession: null,
  puncVocab: null,
  puncList: null,
  tokens: null,
  cmvn: null,
  config: { ...DEFAULT_CONFIG },
  audioData: null,
  audioInfo: null,
};

const els = {
  modelStatus: document.getElementById("modelStatus"),
  audioInput: document.getElementById("audioInput"),
  loadModel: document.getElementById("loadModel"),
  runAsr: document.getElementById("runAsr"),
  log: document.getElementById("log"),
  resultText: document.getElementById("resultText"),
};

function log(message) {
  els.log.textContent = `${message}\n${els.log.textContent}`;
}

function setStatus(message) {
  els.modelStatus.textContent = message;
}

function formatError(err) {
  if (err instanceof Error) {
    return `${err.name}: ${err.message}${err.stack ? `\n${err.stack}` : ""}`;
  }
  if (typeof err === "object") {
    try {
      return JSON.stringify(err);
    } catch (jsonErr) {
      return String(err);
    }
  }
  return String(err);
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
  setStatus("Loading model...");
  await ensureOrtLoaded(false);
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8))
    : 1;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  ort.env.wasm.proxy = false;
  state.session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  state.sessionProvider = "wasm";
  log("Active provider: wasm");

  await Promise.all([
    loadResources(),
    loadPuncResources(),
    (async () => {
      state.puncSession = await ort.InferenceSession.create(PUNC_MODEL_URL, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })(),
  ]);
  setStatus("Model loaded");
  if (state.session?.inputNames?.length) {
    log(`Inputs: ${state.session.inputNames.join(", ")}`);
  }
  if (state.session?.outputNames?.length) {
    log(`Outputs: ${state.session.outputNames.join(", ")}`);
  }
  if (state.puncSession?.inputNames?.length) {
    log(`Punc inputs: ${state.puncSession.inputNames.join(", ")}`);
  }
  els.runAsr.disabled = !state.audioData;
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

async function decodeAudioFile(file) {
  const arrayBuffer = await file.arrayBuffer();
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
  if (!state.session || !state.audioData) return;
  els.runAsr.disabled = true;
  els.resultText.textContent = "";
  setStatus("Running ASR...");

  let stage = "start";
  try {
    const chunkSamples = Math.floor(
      state.config.sampleRate * state.config.maxChunkSeconds
    );
    const totalSamples = state.audioData.length;
    const chunks = Math.ceil(totalSamples / chunkSamples);
    log(`Chunks: ${chunks} (max ${state.config.maxChunkSeconds}s each)`);

    let combined = "";
    for (let i = 0; i < chunks; i++) {
      const start = i * chunkSamples;
      const end = Math.min(totalSamples, start + chunkSamples);
      const slice = state.audioData.subarray(start, end);
      stage = `prepare-input-${i + 1}/${chunks}`;
      const { speech, speechLengths } = prepareInput(
        slice,
        state.config,
        state.cmvn
      );

      log(`Chunk ${i + 1}: frames=${speech.dims[1]}, dim=${speech.dims[2]}`);
      stage = `onnx-run-${i + 1}/${chunks}`;
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
      log(
        `Chunk ${i + 1}: logits=${logits.dims.join("x")}, token_num=${String(
          ArrayBuffer.isView(tokenNum.data) ? tokenNum.data[0] : tokenNum.data
        )}`
      );

      stage = `decode-${i + 1}/${chunks}`;
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
      els.resultText.textContent = combined || "(empty)";
    }

    setStatus("Done");
  } catch (err) {
    log(`Stage: ${stage}`);
    log(`Error type: ${typeof err}`);
    log(formatError(err));
    setStatus("ASR failed");
  } finally {
    els.runAsr.disabled = false;
  }
}

els.loadModel.addEventListener("click", async () => {
  try {
    await loadModel();
  } catch (err) {
    log(String(err));
    setStatus("Model load failed");
  }
});

els.audioInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  setStatus("Decoding audio...");
  try {
    const decoded = await decodeAudioFile(file);
    state.audioData = decoded.samples;
    state.audioInfo = decoded;
    log(
      `Loaded ${file.name} (${decoded.duration.toFixed(2)}s @ ${decoded.sampleRate}Hz)`
    );
    setStatus(state.session ? "Ready" : "Audio loaded");
    els.runAsr.disabled = !state.session;
  } catch (err) {
    log(String(err));
    setStatus("Audio decode failed");
  }
});

els.runAsr.addEventListener("click", async () => {
  await runAsr();
});
