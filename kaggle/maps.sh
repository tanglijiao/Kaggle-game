#!/usr/bin/env bash
# Inference-only run: probability maps for the test set and the public UMUD expert-labelled sets (OSF xbawc).
# Models: MODELS_DIR (default: any models/ folder under /kaggle/input, e.g. an attached earlier run), else train.
set -euo pipefail
export OPENCV_LOG_LEVEL=ERROR
ENC=${ENC:-resnet34}
pip install -q segmentation-models-pytorch
nvidia-smi -L || true

EXT=/kaggle/tmp/ext
mkdir -p "$EXT" && cd "$EXT"
get() { [ -f "$2" ] || wget -q -O "$2" "https://osf.io/download/$1/"; unzip -q -o "$2" -d "${2%.zip}"; }
get hb5xj bench.zip       # 35 images, 7 expert raters (+ DLTrack, SMA), px/cm given
get vqb7h gm.zip          # 168 Telemed GM calf-raise frames, 3 raters (PA deg, FL px)
get zcuwk young.zip       # VL, Esaote: annotated (.tif, red lines) + raw (.tiff)
get 6ga94 old.zip
mkdir -p anno_raw
find young old -name "*.tiff" | while read f; do
  d=$(echo "$f" | sed -E 's#.*/(NA[YO]_[0-9]+)/(timepoint[0-9]+)/(.*)\.tiff#\1__\2__\3#'); cp "$f" "anno_raw/$d.tiff"; done
ls anno_raw | wc -l
cd /kaggle/working/repo

MODELS_DIR=${MODELS_DIR:-$(dirname "$(find /kaggle/input -name "apo_unet_${ENC}_512x768*.pt" | head -1)" 2>/dev/null || true)}
if [ -z "$MODELS_DIR" ] || [ ! -f "$MODELS_DIR/apo_unet_${ENC}_512x768.pt" ]; then
  echo "no models found -> training"
  python -u scripts/02_train_seg.py --task apo  --encoder "$ENC" --epochs 30 --bs 16
  python -u scripts/02_train_seg.py --task fasc --encoder "$ENC" --epochs 25 --bs 16
  MODELS_DIR=/kaggle/working/models
fi
echo "models: $MODELS_DIR"; ls -la "$MODELS_DIR"
python -u scripts/05_infer_maps.py \
  --apo "$MODELS_DIR/apo_unet_${ENC}_512x768.pt" --fasc "$MODELS_DIR/fasc_unet_${ENC}_512x768.pt" \
  --set test= --set bench="$EXT/bench" --set gm="$EXT/gm" --set anno="$EXT/anno_raw" \
  --out /kaggle/working/maps
cd /kaggle/working && zip -q -r maps.zip maps && rm -rf maps repo && ls -la /kaggle/working
