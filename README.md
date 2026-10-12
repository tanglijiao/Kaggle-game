# UMUD Challenge — automated muscle architecture from ultrasound

Code for the Kaggle competition
[UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data):
estimate **pennation angle (PA, °)**, **fascicle length (FL, mm)** and **muscle thickness (MT, mm)**
from single B-mode ultrasound images.

## Approach (work in progress)

1. **Scale detection** — read the vertical depth ruler to get px/mm (square pixels assumed, as stated by the organisers).
2. **Segmentation** — aponeuroses and fascicle fragments with U-Net-style models trained on the provided masks.
3. **Geometry** — fit the two most superficial aponeuroses, fit fascicle fragments, extrapolate between aponeuroses,
   compute PA at the deep aponeurosis, FL along the fascicle, MT perpendicular at three positions; robust aggregation.
4. **Post-processing** — temporal smoothing across consecutive video frames.

## Repository layout

```
src/umud/        library code (paths, scale, geometry, models)
scripts/         numbered pipeline steps (00_explore.py, ...)
kaggle/          notebook launcher used to run the pipeline on Kaggle GPUs
```

## Reproduce

```bash
pip install -r requirements.txt
export UMUD_DATA=/path/to/competition/data   # folder containing sample_submission.csv
python scripts/00_explore.py --out outputs/explore
```

On Kaggle: attach the competition data, enable Internet, and run `kaggle/launcher.py` in a notebook cell.

## Data

Competition data is licensed CC BY-NC-SA 4.0 by the organisers and is not redistributed here.

## License

MIT — see [LICENSE](LICENSE).
