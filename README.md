# WebCore

WebCore is a browser-first AI toolbox that bundles super-resolution (SR), text-to-speech (TTS), and offline ASR with on-device inference.

Website demo: https://www.webcore.xin


## Features

- SR with WebGPU-first execution and WASM fallback
- TTS with model and voice caching plus download progress
- Offline ASR with upload/recording flow and punctuation restoration
- Client-side model caching via IndexedDB

## Project Structure

- `index.html` - main UI
- `sr.js` - super-resolution logic
- `tts.js` - TTS logic
- `asr.js` - ASR logic
- `styles.css` - shared styles
- `ASR/`, `ESRGAN/`, `TTS/` - model assets and references

## Usage

Serve the folder with any static server, then open in a modern browser (Chrome recommended). WebGPU support improves SR performance.

Example with Python:

```bash
python -m http.server 8000
```

Then visit `http://127.0.0.1:8000/`.

## Notes

- Large models are fetched on demand and cached in the browser.
- For best WASM performance, enable COOP/COEP on your server.

## Acknowledgements

- https://github.com/hexgrad/kokoro
- https://huggingface.co/ai-forever/Real-ESRGAN
- https://github.com/modelscope/FunASR

## License

See `LICENSE`.
