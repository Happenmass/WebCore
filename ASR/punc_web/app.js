const MODEL_URL =
  "../manyeyes/alicttransformerpunc-zh-en-mge-int8-onnx/model.int8.onnx";
const TOKENS_URL =
  "../manyeyes/alicttransformerpunc-zh-en-mge-int8-onnx/tokens.txt";
const PUNC_URL = "../manyeyes/alicttransformerpunc-zh-en-mge-int8-onnx/punc.json";

const state = {
  session: null,
  ortBuild: null,
  ortLoading: null,
  vocab: null,
  puncList: null,
};

const els = {
  modelStatus: document.getElementById("modelStatus"),
  textInput: document.getElementById("textInput"),
  loadModel: document.getElementById("loadModel"),
  runPunc: document.getElementById("runPunc"),
  log: document.getElementById("log"),
  resultText: document.getElementById("resultText"),
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

async function ensureOrtLoaded() {
  if (window.ort && state.ortBuild) {
    return;
  }
  if (state.ortLoading) {
    await state.ortLoading;
    return;
  }
  const base = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  const wasmUrl = `${base}ort.min.js`;
  state.ortLoading = (async () => {
    await loadOrtScript(wasmUrl);
    state.ortBuild = "wasm";
  })();
  await state.ortLoading;
}

function normalizeText(text) {
  return text.replace(/[，。？！、]/g, " ").trim();
}

function tokenize(text) {
  const tokens = [];
  const regex = /[A-Za-z0-9]+|[\u4e00-\u9fff]/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    tokens.push(match[0]);
  }
  return tokens;
}

function buildVocab(tokenLines) {
  const vocab = new Map();
  tokenLines.forEach((token, index) => {
    if (!token) return;
    vocab.set(token, index);
  });
  return vocab;
}

async function loadResources() {
  const [tokensText, punc] = await Promise.all([
    fetch(TOKENS_URL).then((res) => res.text()),
    fetch(PUNC_URL).then((res) => res.json()),
  ]);
  const lines = tokensText
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  state.vocab = buildVocab(lines);
  state.puncList = punc.punc_list;
}

async function loadModel() {
  if (state.session) return;
  setStatus("Loading model...");
  await ensureOrtLoaded();
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

  await loadResources();
  setStatus("Model loaded (wasm)");
  log(`Inputs: ${state.session.inputNames.join(", ")}`);
  log(`Outputs: ${state.session.outputNames.join(", ")}`);
  els.runPunc.disabled = false;
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

function joinTokens(tokens, puncIds, puncList) {
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

async function runPunc() {
  const text = els.textInput.value;
  if (!text.trim()) {
    els.resultText.textContent = "";
    return;
  }
  const cleaned = normalizeText(text);
  const tokens = tokenize(cleaned);
  if (!tokens.length) {
    els.resultText.textContent = "";
    return;
  }

  const ids = mapTokensToIds(tokens, state.vocab);
  const inputIds = new Int32Array(ids);
  const inputTensor = new ort.Tensor("int32", inputIds, [1, ids.length]);
  const lenTensor = new ort.Tensor("int32", new Int32Array([ids.length]), [1]);

  const outputs = await state.session.run({
    inputs: inputTensor,
    text_lengths: lenTensor,
  });
  const logits = outputs.logits;
  const length = logits.dims[1];
  const puncSize = logits.dims[2];
  const puncIds = argmaxPunc(logits.data, length, puncSize);
  const restored = joinTokens(tokens, puncIds, state.puncList);
  els.resultText.textContent = restored;
  log(`Tokens: ${tokens.length}, punc classes: ${puncSize}`);
}

els.loadModel.addEventListener("click", async () => {
  try {
    await loadModel();
  } catch (err) {
    log(String(err));
    setStatus("Model load failed");
  }
});

els.runPunc.addEventListener("click", async () => {
  try {
    await runPunc();
  } catch (err) {
    log(String(err));
    setStatus("Punctuation failed");
  }
});
