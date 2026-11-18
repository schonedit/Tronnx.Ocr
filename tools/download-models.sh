#!/usr/bin/env bash
# Letölti a PP-OCRv5 ONNX modelleket a Tronnx.Ocr.Models.Latin projekt models/ mappájába,
# és SHA-256 ellenőrzőösszeggel ellenőrzi őket. (macOS régi bash 3.2-vel is fut.)
# Forrás: a PaddleOCR (Apache 2.0) modelljeinek ONNX-exportja. Javaslat: töltsd fel ugyanezeket
# a saját GitHub-repód egy "models-v1" release-ébe, és írd át a BASE_URL-t, hogy ne függj harmadik féltől.
set -euo pipefail

BASE_URL="${TRONNX_MODELS_URL:-https://github.com/GreatV/oar-ocr/releases/download/v0.3.0}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/src/Tronnx.Ocr.Models.Latin/models"
mkdir -p "$DEST"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

for entry in \
  "pp-ocrv5_mobile_det.onnx:1eb7b4f7ab657ebd1c66d5f79bca7497f29768a2e3c15e52daecbba1a8e4a039" \
  "latin_pp-ocrv5_mobile_rec.onnx:e3a6bfeea1c8a01d6fccfd480a0bd363fd907f8c65931e228bb2736f5c3e142f" \
  "ppocrv5_latin_dict.txt:ccbcc45730b3fbbd9050c5bc74db6a99067141ef1035e3d14889a84a6b9b1aff"
do
  f="${entry%%:*}"
  expected="${entry##*:}"
  target="$DEST/$f"
  if [[ ! -f "$target" ]]; then
    echo "Downloading $f ..."
    curl -fsSL -o "$target" "$BASE_URL/$f"
  fi
  actual="$(sha256 "$target")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $f (got $actual)" >&2
    rm -f "$target"
    exit 1
  fi
  echo "OK  $f"
done
