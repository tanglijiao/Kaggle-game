#!/usr/bin/env bash
# Full baseline run on Kaggle (GPU): train both segmenters, then predict + measure the test set.
set -euo pipefail
export OPENCV_LOG_LEVEL=ERROR
EPOCHS_APO=${EPOCHS_APO:-30}
EPOCHS_FASC=${EPOCHS_FASC:-25}
ENC=${ENC:-resnet34}
pip install -q segmentation-models-pytorch
nvidia-smi -L
python -u scripts/02_train_seg.py --task apo  --encoder "$ENC" --epochs "$EPOCHS_APO"  --bs 16
python -u scripts/02_train_seg.py --task fasc --encoder "$ENC" --epochs "$EPOCHS_FASC" --bs 16
python -u scripts/03_predict.py \
  --apo  /kaggle/working/models/apo_unet_${ENC}_512x768.pt \
  --fasc /kaggle/working/models/fasc_unet_${ENC}_512x768.pt \
  --out /kaggle/working --save-prob
python -u scripts/04_postprocess.py --smooth 0.5
