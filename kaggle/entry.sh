#!/usr/bin/env bash
# The Kaggle notebook cell only clones the repo and runs this file, so switching jobs = editing this line + git push.
set -euo pipefail
ENC=resnet50 bash kaggle/maps.sh
