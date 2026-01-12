#!/usr/bin/env python
# Simple local G2P API for Kokoro (Chinese + English mix).

from flask import Flask, jsonify, request

try:
    from kokoro import KPipeline
except Exception as exc:
    raise SystemExit(
        "Missing dependency: kokoro. Install with "
        "`pip install kokoro>=0.8.1 \"misaki[zh]>=0.8.1\"`."
    ) from exc


app = Flask(__name__)


def build_pipeline(repo_id):
    en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)

    def en_callable(text):
        return next(en_pipeline(text)).phonemes

    return KPipeline(
        lang_code="z",
        repo_id=repo_id,
        model=False,
        en_callable=en_callable,
    )


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "https://www.webcore.xin"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/g2p", methods=["GET", "POST", "OPTIONS"])
def g2p():
    if request.method == "OPTIONS":
        return ("", 204)
    text = request.values.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400

    repo_id = request.values.get("repo_id", "hexgrad/Kokoro-82M-v1.1-zh")
    pipeline = build_pipeline(repo_id)
    phonemes = []
    segments = []
    for result in pipeline(text):
        if result.phonemes:
            phonemes.append(result.phonemes)
            segments.append(result.phonemes)
    if not phonemes:
        return jsonify({"error": "No phonemes produced"}), 400
    return jsonify({"phonemes": "".join(phonemes), "segments": segments})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False)
