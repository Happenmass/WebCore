const MODEL_URL = "../kokoro_fp32.onnx";
const CONFIG_URL = "../hexgrad/Kokoro-82M-v1.1-zh/config.json";
const SAMPLE_RATE = 24000;
const VOICE_JSON_BASE = "./voices";
const PHONEMIZE_URL =
  "https://cdn.jsdelivr.net/npm/phonemize@1.1.0/dist/index.mjs";
const G2P_API_URL = "http://127.0.0.1:5005/g2p";

const state = {
  session: null,
  vocab: null,
  contextLength: 512,
  voicePack: null,
  audioBuffer: null,
  audioCtx: null,
  phonemize: null,
};

const els = {
  modelStatus: document.getElementById("modelStatus"),
  textInput: document.getElementById("textInput"),
  voiceSelect: document.getElementById("voiceSelect"),
  voiceFile: document.getElementById("voiceFile"),
  refIndex: document.getElementById("refIndex"),
  speed: document.getElementById("speed"),
  speedValue: document.getElementById("speedValue"),
  loadModel: document.getElementById("loadModel"),
  synthesize: document.getElementById("synthesize"),
  playBtn: document.getElementById("playBtn"),
  downloadBtn: document.getElementById("downloadBtn"),
  log: document.getElementById("log"),
};

function log(message) {
  els.log.textContent = `${message}\n${els.log.textContent}`;
}

function setStatus(message) {
  els.modelStatus.textContent = message;
}

function normalizePhonemes(text) {
  return text.trim();
}

async function g2pFromLocal(text) {
  const url = new URL(G2P_API_URL);
  url.searchParams.set("text", text);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error("Local G2P API failed.");
  }
  const data = await res.json();
  if (!data.phonemes) {
    throw new Error("Local G2P API returned empty phonemes.");
  }
  return data.phonemes;
}

const INITIALS = [
  "zh",
  "ch",
  "sh",
  "b",
  "p",
  "m",
  "f",
  "d",
  "t",
  "n",
  "l",
  "g",
  "k",
  "h",
  "j",
  "q",
  "x",
  "r",
  "z",
  "c",
  "s",
  "y",
  "w",
];

const ZH_MAP = {
  b: "ㄅ",
  p: "ㄆ",
  m: "ㄇ",
  f: "ㄈ",
  d: "ㄉ",
  t: "ㄊ",
  n: "ㄋ",
  l: "ㄌ",
  g: "ㄍ",
  k: "ㄎ",
  h: "ㄏ",
  j: "ㄐ",
  q: "ㄑ",
  x: "ㄒ",
  zh: "ㄓ",
  ch: "ㄔ",
  sh: "ㄕ",
  r: "ㄖ",
  z: "ㄗ",
  c: "ㄘ",
  s: "ㄙ",
  a: "ㄚ",
  o: "ㄛ",
  e: "ㄜ",
  ie: "ㄝ",
  ai: "ㄞ",
  ei: "ㄟ",
  ao: "ㄠ",
  ou: "ㄡ",
  an: "ㄢ",
  en: "ㄣ",
  ang: "ㄤ",
  eng: "ㄥ",
  er: "ㄦ",
  i: "ㄧ",
  u: "ㄨ",
  v: "ㄩ",
  ii: "ㄭ",
  iii: "十",
  ve: "月",
  ia: "压",
  ian: "言",
  iang: "阳",
  iao: "要",
  in: "阴",
  ing: "应",
  iong: "用",
  iou: "又",
  ong: "中",
  ua: "穵",
  uai: "外",
  uan: "万",
  uang: "王",
  uei: "为",
  uen: "文",
  ueng: "瓮",
  uo: "我",
  van: "元",
  vn: "云",
  "1": "1",
  "2": "2",
  "3": "3",
  "4": "4",
  "5": "5",
  ";": ";",
  ":": ":",
  ",": ",",
  ".": ".",
  "!": "!",
  "?": "?",
  "/": "/",
  "—": "—",
  "…": "…",
  '"': '"',
  "(": "(",
  ")": ")",
  "“": "“",
  "”": "”",
  " ": " ",
};

function mapPunctuation(text) {
  return text
    .replace(/、/g, ", ")
    .replace(/，/g, ", ")
    .replace(/。/g, ". ")
    .replace(/．/g, ". ")
    .replace(/！/g, "! ")
    .replace(/：/g, ": ")
    .replace(/；/g, "; ")
    .replace(/？/g, "? ")
    .replace(/«/g, " “")
    .replace(/»/g, "” ")
    .replace(/《/g, " “")
    .replace(/》/g, "” ")
    .replace(/「/g, " “")
    .replace(/」/g, "” ")
    .replace(/【/g, " “")
    .replace(/】/g, "” ")
    .replace(/（/g, " (")
    .replace(/）/g, ") ");
}

function splitPinyin(py) {
  const match = py.match(/^([a-z]+)([1-5])$/i);
  if (!match) {
    return null;
  }
  const base = match[1].toLowerCase();
  const tone = match[2];
  let initial = "";
  for (const c of INITIALS) {
    if (base.startsWith(c)) {
      initial = c;
      break;
    }
  }
  let final = base.slice(initial.length);
  if (final.startsWith("i") && ["z", "c", "s"].includes(initial)) {
    final = "ii" + final.slice(1);
  } else if (
    final.startsWith("i") &&
    ["zh", "ch", "sh", "r"].includes(initial)
  ) {
    final = "iii" + final.slice(1);
  }
  if (final.startsWith("u:")) {
    final = "v" + final.slice(2);
  }
  if (final.startsWith("ü")) {
    final = "v" + final.slice(1);
  }
  return { initial, final, tone };
}

function zhSyllableToPhonemes(py) {
  const parsed = splitPinyin(py);
  if (!parsed) {
    return "";
  }
  const parts = [];
  if (parsed.initial) {
    parts.push(parsed.initial);
  }
  if (parsed.final) {
    parts.push(`${parsed.final}${parsed.tone}`);
  } else {
    parts.push(parsed.tone);
  }
  const joined = parts.join("_");
  const tokens = joined.replace(/(?=\d)/g, "_").split("_");
  return tokens.map((t) => ZH_MAP[t] || "").join("");
}

function zhToPhonemes(text) {
  const pinyinLib = window.pinyinPro || window.pinyin || null;
  const pinyinFn = pinyinLib?.pinyin;
  if (!pinyinFn) {
    throw new Error("pinyin-pro not loaded.");
  }
  const pyArray = pinyinFn(text, {
    toneType: "num",
    type: "array",
    v: "v",
  });
  if (!pyArray) {
    throw new Error("pinyin-pro not loaded.");
  }
  const phones = pyArray.map((py) => zhSyllableToPhonemes(py));
  return phones.join("/");
}

async function enToPhonemes(text) {
  await ensurePhonemize();
  const fn = state.phonemize;
  if (!fn) {
    return text.replace(/\s+/g, " ");
  }
  const result = fn(text);
  return String(result).replace(/\s+/g, " ").trim();
}

async function textToPhonemes(text) {
  const mapped = mapPunctuation(text);
  const segments =
    mapped.match(/[\u4e00-\u9fff]+|[A-Za-z'-]+|[^A-Za-z\u4e00-\u9fff]+/g) ||
    [];
  let result = "";
  for (const seg of segments) {
    if (/[\u4e00-\u9fff]/.test(seg)) {
      result += zhToPhonemes(seg);
      continue;
    }
    if (/[A-Za-z]/.test(seg)) {
      result += await enToPhonemes(seg);
      continue;
    }
    result += seg;
  }
  return result.trim();
}

function phonemesToIds(phonemes) {
  const ids = [];
  for (const ch of phonemes) {
    const id = state.vocab[ch];
    if (id !== undefined && id !== null) {
      ids.push(id);
    }
  }
  if (ids.length + 2 > state.contextLength) {
    throw new Error(`Input too long: ${ids.length + 2} > ${state.contextLength}`);
  }
  return [0, ...ids, 0];
}

async function loadConfig() {
  const res = await fetch(CONFIG_URL);
  if (!res.ok) {
    throw new Error("Failed to load config.json");
  }
  const config = await res.json();
  state.vocab = config.vocab;
  state.contextLength = config.plbert?.max_position_embeddings || 512;
}

async function ensurePhonemize() {
  if (state.phonemize) {
    return;
  }
  log("Loading phonemize...");
  let mod;
  try {
    mod = await import(PHONEMIZE_URL);
  } catch (err) {
    throw new Error("Failed to load phonemize.");
  }
  const fn = mod.phonemize || mod.toIPA || mod.default;
  if (typeof fn !== "function") {
    throw new Error("phonemize API not found.");
  }
  state.phonemize = fn;
}

async function loadVoicePackFromFile(file) {
  const text = await file.text();
  const data = JSON.parse(text);
  if (!Array.isArray(data)) {
    throw new Error("Voice pack JSON must be an array.");
  }
  state.voicePack = data;
  log("Loaded voice pack from file.");
}

async function loadVoicePackFromUrl(voiceId) {
  const url = `${VOICE_JSON_BASE}/${voiceId}.json`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load voice pack: ${url}`);
  }
  state.voicePack = await res.json();
  log(`Loaded voice pack: ${voiceId}`);
}

function selectRefS(phonemeLength) {
  if (!state.voicePack) {
    throw new Error("Voice pack not loaded.");
  }
  const index = Number(els.refIndex.value);
  if (Number.isInteger(index) && index >= 0) {
    return state.voicePack[index];
  }
  return state.voicePack[phonemeLength - 1];
}

function normalizeRefS(refS) {
  if (Array.isArray(refS) && refS.length === 1 && Array.isArray(refS[0])) {
    return refS[0];
  }
  return refS;
}

function toOrtTensorInt64(values, dims) {
  const bigint = BigInt64Array.from(values.map((v) => BigInt(v)));
  return new ort.Tensor("int64", bigint, dims);
}

function toOrtTensorFloat(values, dims) {
  return new ort.Tensor("float32", new Float32Array(values), dims);
}

function buildWavBlob(samples) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, s, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function ensureSession() {
  if (state.session) {
    return;
  }
  setStatus("加载模型中...");
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency || 4);
  ort.env.wasm.simd = true;
  await loadConfig();
  const providers = [];
  if (ort.env.webgpu && ort.env.webgpu.isSupported) {
    providers.push("webgpu");
  }
  providers.push("wasm");
  state.session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: providers,
  });
  setStatus("模型已加载");
  els.synthesize.disabled = false;
}

async function synthesize() {
  try {
    const raw = normalizePhonemes(els.textInput.value);
    if (!raw) {
      alert("请输入文本或 phonemes");
      return;
    }
    let text = "";
    try {
      text = await g2pFromLocal(raw);
      log("G2P: local API");
    } catch (err) {
      log("G2P: local API failed, fallback to frontend.");
      text = await textToPhonemes(raw);
    }
    if (!text) {
      alert("请输入 phonemes");
      return;
    }
    if (!state.session) {
      await ensureSession();
    }
    if (!state.voicePack) {
      await loadVoicePackFromUrl(els.voiceSelect.value);
    }

    log(`Phonemes: ${text}`);
    const inputIds = phonemesToIds(text);
    const refS = normalizeRefS(selectRefS(text.length));
    const speed = Number(els.speed.value);

    const feeds = {
      input_ids: toOrtTensorInt64(inputIds, [1, inputIds.length]),
      ref_s: toOrtTensorFloat(refS, [1, refS.length]),
      speed: toOrtTensorFloat([speed], []),
    };

    log("Running inference...");
    const results = await state.session.run(feeds);
    const waveform = results.waveform?.data || results.output_0?.data;
    if (!waveform) {
      throw new Error("No waveform output.");
    }

    const audioArray = Float32Array.from(waveform);
    state.audioBuffer = audioArray;
    els.playBtn.disabled = false;
    els.downloadBtn.disabled = false;
    log(`Done. Samples: ${audioArray.length}`);
  } catch (err) {
    console.error(err);
    log(`Error: ${err.message}`);
  }
}

function playAudio() {
  if (!state.audioBuffer) return;
  if (!state.audioCtx) {
    state.audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  }
  const buffer = state.audioCtx.createBuffer(
    1,
    state.audioBuffer.length,
    SAMPLE_RATE
  );
  buffer.getChannelData(0).set(state.audioBuffer);
  const source = state.audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(state.audioCtx.destination);
  source.start();
}

function downloadAudio() {
  if (!state.audioBuffer) return;
  const blob = buildWavBlob(state.audioBuffer);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "kokoro_onnx.wav";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

els.speed.addEventListener("input", () => {
  els.speedValue.textContent = Number(els.speed.value).toFixed(2);
});

els.voiceFile.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file) {
    await loadVoicePackFromFile(file);
  }
});

els.loadModel.addEventListener("click", ensureSession);
els.synthesize.addEventListener("click", synthesize);
els.playBtn.addEventListener("click", playAudio);
els.downloadBtn.addEventListener("click", downloadAudio);
