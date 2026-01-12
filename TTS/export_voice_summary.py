#!/usr/bin/env python3
"""Export grouped voice summary from analysis JSONL."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def normalize_description(text: str) -> str:
    return text.strip()


def map_gender(value: str) -> Optional[str]:
    if not value:
        return None
    lower = value.strip().lower()
    if lower in ("male", "m", "man", "男"):
        return "男"
    if lower in ("female", "f", "woman", "女"):
        return "女"
    return None


def dedupe_and_insert(items: List[Dict[str, str]], desc: str, voice: str) -> None:
    for idx, item in enumerate(items):
        existing = item["description"]
        if desc == existing or desc in existing or existing in desc:
            if len(desc) > len(existing):
                items[idx] = {"description": desc, "voice": voice}
            return
    items.append({"description": desc, "voice": voice})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export grouped voice summary JSON from analysis JSONL."
    )
    parser.add_argument(
        "--analysis-in",
        default="voice_analysis.jsonl",
        help="Input JSONL produced by analyze_voices.py.",
    )
    parser.add_argument(
        "--output",
        default="voice_summary.json",
        help="Output JSON with grouped voices.",
    )
    args = parser.parse_args()

    analysis_path = Path(args.analysis_in)
    if not analysis_path.is_file():
        raise SystemExit(f"Missing analysis file: {analysis_path}")

    grouped: Dict[str, List[Dict[str, str]]] = {"男": [], "女": []}

    with analysis_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            analysis = record.get("analysis")
            if not isinstance(analysis, dict):
                continue
            gender = map_gender(analysis.get("gender", ""))
            if gender is None:
                continue
            description = analysis.get("description", "")
            if not isinstance(description, str):
                continue
            description = normalize_description(description)
            if not description:
                continue
            voice = record.get("voice") or record.get("voice_id") or ""
            if not voice:
                continue

            dedupe_and_insert(grouped[gender], description, voice)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
