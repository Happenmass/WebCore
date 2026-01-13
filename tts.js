(() => {
const APP_VERSION = typeof window !== "undefined" && window.APP_VERSION ? window.APP_VERSION : "";

function withVersion(url) {
  if (!APP_VERSION) return url;
  const suffix = `v=${encodeURIComponent(APP_VERSION)}`;
  return url.includes("?") ? `${url}&${suffix}` : `${url}?${suffix}`;
}

function getVoiceCacheKey(voiceId) {
  if (!APP_VERSION) return voiceId;
  return `${voiceId}?v=${APP_VERSION}`;
}

const TTS_MODEL_BASE = `https://static.webcore.xin`;
const TTS_MODEL_URL = withVersion(`${TTS_MODEL_BASE}/kokoro/kokoro_fp32.onnx`);
const TTS_VOICE_BASE_URL = `${TTS_MODEL_BASE}/voices`;
const TTS_VOICE_SUMMARY_URL = withVersion("./voice_summary.json");
const TTS_CONFIG_URL = withVersion("./config.json");
const TTS_PHONEMIZE_URL = "https://cdn.jsdelivr.net/npm/phonemize@1.1.0/dist/index.mjs";
const TTS_G2P_REMOTE_URL = "https://www.webcore.xin/g2p";
const TTS_SAMPLE_RATE = 24000;
const TTS_ORG = "tts";
const MODEL_CACHE_DB = "tts-model-cache-v2";
const MODEL_CACHE_STORE = "models";
const VOICE_CACHE_STORE = "voices";

const ttsState = {
  summary: null,
  voices: {},
  gender: "女",
  voice: null,
  voicePack: null,
  session: null,
  vocab: null,
  contextLength: 512,
  audioBuffer: null,
  audioCtx: null,
  phonemize: null,
  ortLoading: null,
  pinyinLoading: null,
};

const i18n = window.I18N;
const t = (key, fallback) => (i18n ? i18n.t(key, fallback) : fallback || key);

const ttsEls = {
  text: document.getElementById("tts-text"),
  gender: document.getElementById("tts-gender"),
  voice: document.getElementById("tts-voice"),
  run: document.getElementById("tts-run"),
  status: document.getElementById("tts-status"),
  play: document.getElementById("tts-play"),
  download: document.getElementById("tts-download"),
  copy: document.getElementById("tts-copy"),
  audio: document.getElementById("tts-audio"),
  modelProgress: document.getElementById("tts-model-progress"),
  modelProgressBar: document.getElementById("tts-model-progress-bar"),
};

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

function setStatus(message, key) {
  if (ttsEls.status) {
    ttsEls.status.textContent = message;
    if (key) {
      ttsEls.status.dataset.statusKey = key;
    }
  }
}

function setModelProgress(value) {
  if (!ttsEls.modelProgress || !ttsEls.modelProgressBar) return;
  const clamped = Math.max(0, Math.min(1, value));
  ttsEls.modelProgress.style.visibility =
    clamped > 0 && clamped < 1 ? "visible" : "hidden";
  ttsEls.modelProgressBar.style.width = `${clamped * 100}%`;
}

function resetOrt(reason) {
  console.log("[TTS] reset ort", reason);
  window.ort = null;
  ttsState.ortLoading = null;
  ttsState.session = null;
}

function setButtonsEnabled(enabled) {
  if (ttsEls.play) ttsEls.play.disabled = !enabled;
  if (ttsEls.download) ttsEls.download.disabled = !enabled;
  if (ttsEls.copy) ttsEls.copy.disabled = !enabled;
  if (ttsEls.audio) ttsEls.audio.disabled = !enabled;
}

function loadScript(url) {
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
      if (!db.objectStoreNames.contains(VOICE_CACHE_STORE)) {
        db.createObjectStore(VOICE_CACHE_STORE);
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

async function readVoiceFromCache(key) {
  const db = await openModelCache();
  if (!db) return null;
  return new Promise((resolve, reject) => {
    const tx = db.transaction(VOICE_CACHE_STORE, "readonly");
    const store = tx.objectStore(VOICE_CACHE_STORE);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
}

async function writeVoiceToCache(key, data) {
  const db = await openModelCache();
  if (!db) return;
  return new Promise((resolve, reject) => {
    const tx = db.transaction(VOICE_CACHE_STORE, "readwrite");
    const store = tx.objectStore(VOICE_CACHE_STORE);
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

async function ensureOrtLoaded() {
  if (window.ort) return;
  if (ttsState.ortLoading) {
    await ttsState.ortLoading;
    return;
  }
  const url = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js";
  ttsState.ortLoading = loadScript(url);
  await ttsState.ortLoading;
}

async function ensurePinyinLoaded() {
  if (window.pinyinPro?.pinyin) return;
  if (ttsState.pinyinLoading) {
    await ttsState.pinyinLoading;
    return;
  }
  const url = "https://cdn.jsdelivr.net/npm/pinyin-pro@3.20.4/dist/index.js";
  ttsState.pinyinLoading = loadScript(url);
  await ttsState.pinyinLoading;
}

function setSelectOptions(select, options) {
  if (!select) return;
  select.innerHTML = "";
  for (const option of options) {
    select.appendChild(option);
  }
}

function buildOption(label, value, selected) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  if (selected) {
    opt.selected = true;
  }
  return opt;
}

function populateGender() {
  if (!ttsState.summary || !ttsEls.gender) return;
  const genders = Object.keys(ttsState.summary);
  const defaultGender = genders.includes("女") ? "女" : genders[0];
  ttsState.gender = defaultGender;
  const options = genders.map((gender) =>
    buildOption(gender, gender, gender === defaultGender)
  );
  setSelectOptions(ttsEls.gender, options);
}

function populateVoiceList() {
  if (!ttsState.summary || !ttsEls.voice) return;
  const list = ttsState.summary[ttsState.gender] || [];
  const defaultVoice = list[0]?.voice || null;
  ttsState.voice = defaultVoice;
  const options = list.map((item, index) =>
    buildOption(item.description, item.voice, index === 0)
  );
  setSelectOptions(ttsEls.voice, options);
  if (defaultVoice) {
    loadVoice(defaultVoice).catch((err) => {
      console.error("Failed to load voice:", err);
    });
  }
}

async function loadVoice(voiceId) {
  if (!voiceId) return;
  if (ttsState.voices[voiceId]) {
    ttsState.voicePack = ttsState.voices[voiceId];
    return;
  }
  const cacheKey = getVoiceCacheKey(voiceId);
  const cached = await readVoiceFromCache(cacheKey);
  if (cached) {
    ttsState.voices[voiceId] = cached;
    ttsState.voicePack = cached;
    return;
  }
  const url = withVersion(`${TTS_VOICE_BASE_URL}/${voiceId}.json`);
  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`Failed to fetch voice ${voiceId}`);
  }
  const data = await response.json();
  ttsState.voices[voiceId] = data;
  ttsState.voicePack = data;
  await writeVoiceToCache(cacheKey, data);
}

async function initVoices() {
  if (!ttsEls.gender || !ttsEls.voice) return;
  const response = await fetch(TTS_VOICE_SUMMARY_URL);
  if (!response.ok) {
    throw new Error("Failed to load voice summary");
  }
  ttsState.summary = await response.json();
  populateGender();
  populateVoiceList();
}

function normalizePhonemes(text) {
  return text.trim();
}

async function g2pFromApi(url, text) {
  const target = new URL(url);
  target.searchParams.set("text", text);
  const res = await fetch(target);
  if (!res.ok) {
    throw new Error("G2P API failed.");
  }
  const data = await res.json();
  if (!data.phonemes) {
    throw new Error("G2P API returned empty phonemes.");
  }
  return data.phonemes;
}

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
  const final = base.slice(initial.length);
  if (!final) {
    return null;
  }
  return { initial, final, tone };
}

function normalizePinyin(py) {
  let fixed = py.toLowerCase().replace("ü", "v").replace("u:", "v");
  if (fixed === "zhi") fixed = "zhiii";
  if (fixed === "chi") fixed = "chiii";
  if (fixed === "shi") fixed = "shiii";
  if (fixed === "zi") fixed = "ziii";
  if (fixed === "ci") fixed = "ciii";
  if (fixed === "si") fixed = "siii";
  if (fixed === "ri") fixed = "riii";
  return fixed;
}

function zhSyllableToPhonemes(syllable) {
  const fixed = normalizePinyin(syllable);
  const parts = splitPinyin(fixed);
  if (!parts) {
    return syllable;
  }
  const initial = parts.initial ? ZH_MAP[parts.initial] || "" : "";
  const final = ZH_MAP[parts.final] || parts.final;
  const tone = ZH_MAP[parts.tone] || parts.tone;
  return `${initial}${final}${tone}`.trim();
}

function zhToPhonemes(text) {
  const pinyinFn = window.pinyinPro?.pinyin;
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

async function ensurePhonemize() {
  if (ttsState.phonemize) {
    return;
  }
  let mod;
  try {
    mod = await import(TTS_PHONEMIZE_URL);
  } catch (err) {
    throw new Error("Failed to load phonemize.");
  }
  const fn = mod.phonemize || mod.toIPA || mod.default;
  if (typeof fn !== "function") {
    throw new Error("phonemize API not found.");
  }
  ttsState.phonemize = fn;
}

async function enToPhonemes(text) {
  await ensurePhonemize();
  const fn = ttsState.phonemize;
  if (!fn) {
    return text.replace(/\s+/g, " ");
  }
  const result = fn(text);
  return String(result).replace(/\s+/g, " ").trim();
}

async function textToPhonemes(text) {
  await ensurePinyinLoaded();
  const mapped = mapPunctuation(text);
  const segments =
    mapped.match(/[\u4e00-\u9fff]+|[A-Za-z'-]+|[^A-Za-z\u4e00-\u9fff]+/g) || [];
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
    const id = ttsState.vocab[ch];
    if (id !== undefined && id !== null) {
      ids.push(id);
    }
  }
  if (ids.length + 2 > ttsState.contextLength) {
    throw new Error(`Input too long: ${ids.length + 2} > ${ttsState.contextLength}`);
  }
  return [0, ...ids, 0];
}

async function loadConfig() {
  const res = await fetch(TTS_CONFIG_URL);
  if (!res.ok) {
    throw new Error("Failed to load config.json");
  }
  const config = await res.json();
  ttsState.vocab = config.vocab;
  ttsState.contextLength = config.plbert?.max_position_embeddings || 512;
}

function selectRefS(phonemeLength) {
  if (!ttsState.voicePack) {
    throw new Error("Voice pack not loaded.");
  }
  return ttsState.voicePack[phonemeLength - 1];
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
  view.setUint32(24, TTS_SAMPLE_RATE, true);
  view.setUint32(28, TTS_SAMPLE_RATE * 2, true);
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

function setAudioSource(samples) {
  if (!ttsEls.audio || !samples) return;
  const blob = buildWavBlob(samples);
  const url = URL.createObjectURL(blob);
  if (ttsEls.audio.dataset.url) {
    URL.revokeObjectURL(ttsEls.audio.dataset.url);
  }
  ttsEls.audio.dataset.url = url;
  ttsEls.audio.src = url;
}

async function ensureSession() {
  if (ttsState.session) return;
  setStatus(
    t("tts_status_model_loading", "模型加载中..."),
    "tts_status_model_loading"
  );
  setModelProgress(0);
  await ensureOrtLoaded();
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency || 4);
  ort.env.wasm.simd = true;
  await loadConfig();
  let modelData = await readModelFromCache(TTS_MODEL_URL);
  if (!modelData) {
    modelData = await fetchModelWithProgress(TTS_MODEL_URL);
    await writeModelToCache(TTS_MODEL_URL, modelData);
  }
  ttsState.session = await ort.InferenceSession.create(modelData, {
    executionProviders: ["wasm"],
  });
  setStatus(t("tts_status_model_loaded", "模型已加载"), "tts_status_model_loaded");
  setModelProgress(1);
}

async function synthesize() {
  if (!ttsEls.text) return;
  const raw = normalizePhonemes(ttsEls.text.value);
  if (!raw) {
    setStatus(t("tts_status_need_text", "请输入文本"), "tts_status_need_text");
    return;
  }
  if (ttsEls.run) ttsEls.run.disabled = true;
  setButtonsEnabled(false);
  try {
    setStatus(t("tts_status_preparing", "准备中..."), "tts_status_preparing");
    if (window.__ortOwner !== TTS_ORG) {
      resetOrt("switch to tts");
      window.__ortOwner = TTS_ORG;
    }
    let text = "";
    try {
      text = await g2pFromApi(TTS_G2P_REMOTE_URL, raw);
    } catch (err) {
      text = await textToPhonemes(raw);
    }
    if (!text) {
      setStatus(t("tts_status_g2p_failed", "G2P 失败"), "tts_status_g2p_failed");
      return;
    }
    await ensureSession();
    if (!ttsState.voicePack && ttsState.voice) {
      await loadVoice(ttsState.voice);
    }

    const inputIds = phonemesToIds(text);
    const refS = normalizeRefS(selectRefS(text.length));
    const speed = 1.0;

    setStatus(
      t("tts_status_synthesizing", "合成中..."),
      "tts_status_synthesizing"
    );
    const feeds = {
      input_ids: toOrtTensorInt64(inputIds, [1, inputIds.length]),
      ref_s: toOrtTensorFloat(refS, [1, refS.length]),
      speed: toOrtTensorFloat([speed], []),
    };

    const results = await ttsState.session.run(feeds);
    const waveform = results.waveform?.data || results.output_0?.data;
    if (!waveform) {
      throw new Error("No waveform output.");
    }
    ttsState.audioBuffer = Float32Array.from(waveform);
    setAudioSource(ttsState.audioBuffer);
    setButtonsEnabled(true);
    setStatus(t("tts_status_success", "合成成功"), "tts_status_success");
  } catch (err) {
    console.error(err);
    setStatus(t("tts_status_failed", "合成失败"), "tts_status_failed");
  } finally {
    if (ttsEls.run) ttsEls.run.disabled = false;
  }
}

function playAudio() {
  if (!ttsState.audioBuffer) return;
  if (!ttsState.audioCtx) {
    ttsState.audioCtx = new AudioContext({ sampleRate: TTS_SAMPLE_RATE });
  }
  const buffer = ttsState.audioCtx.createBuffer(
    1,
    ttsState.audioBuffer.length,
    TTS_SAMPLE_RATE
  );
  buffer.getChannelData(0).set(ttsState.audioBuffer);
  const source = ttsState.audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(ttsState.audioCtx.destination);
  source.start();
}

function downloadAudio() {
  if (!ttsState.audioBuffer) return;
  const blob = buildWavBlob(ttsState.audioBuffer);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "kokoro_onnx.wav";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function copyBase64() {
  if (!ttsState.audioBuffer) return;
  const blob = buildWavBlob(ttsState.audioBuffer);
  const reader = new FileReader();
  reader.onload = () => {
    const result = String(reader.result || "");
    const base64 = result.split(",")[1] || "";
    navigator.clipboard.writeText(base64).then(
      () => setStatus(t("tts_status_copy_success", "Base64 已复制"), "tts_status_copy_success"),
      () => setStatus(t("tts_status_copy_failed", "复制失败"), "tts_status_copy_failed")
    );
  };
  reader.readAsDataURL(blob);
}

if (ttsEls.gender) {
  ttsEls.gender.addEventListener("change", (event) => {
    ttsState.gender = event.target.value;
    populateVoiceList();
  });
}

if (ttsEls.voice) {
  ttsEls.voice.addEventListener("change", (event) => {
    const voiceId = event.target.value;
    ttsState.voice = voiceId;
    loadVoice(voiceId).catch((err) => {
      console.error("Failed to load voice:", err);
    });
  });
}

if (ttsEls.run) {
  ttsEls.run.addEventListener("click", () => {
    synthesize();
  });
}

if (ttsEls.play) {
  ttsEls.play.addEventListener("click", playAudio);
}

if (ttsEls.download) {
  ttsEls.download.addEventListener("click", downloadAudio);
}

if (ttsEls.copy) {
  ttsEls.copy.addEventListener("click", copyBase64);
}

setButtonsEnabled(false);
setModelProgress(0);
initVoices().catch((err) => {
  console.error(err);
  setStatus(
    t("tts_status_voice_list_failed", "音色列表加载失败"),
    "tts_status_voice_list_failed"
  );
});
})();
