# Findings log

## 2026-10-12 — Stage 0 data inventory (Kaggle, scripts/00_explore.py @ 7dba59d)

Data root on Kaggle: `/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data`
Folders: `apo_imgs_v1`, `apo_masks_v1`, `fasc_imgs_v1`, `fasc_masks_v1`, `test_images_v2`, `sample_submission.csv`.

| split | n | sizes (WxH: n) | format |
|---|---|---|---|
| apo_imgs | 1048 | 1200x800: 732, 512x512: 188, 800x652: 76, 660x556: 23, 800x600: 15, 1234x688: 13, 1074x688: 1 | tif, RGB (1 grey) |
| apo_masks | 1048 | 1152x864: 474, 660x556: 276, 512x512: 188, 800x652: 77, 1234x688: 16, 800x600: 15, ... | tif, 1 ch |
| fasc_imgs | 2761 | 1200x800: 1714, 1640x1080: 645, 760x500: 216, 512x512: 79, 800x600: 65, 800x652: 42 | tif, RGB |
| fasc_masks | 2761 | 996x556: 741, 1640x1080: 645, 1152x864: 426, 760x500: 216, 660x556: 169, 1340x810: 121, 1088x644: 121, 1196x768: 101, ... | tif, 1 ch |
| test_images | 309 | 1200x800: 239, 1088x644: 50, 1069x853: 11, ~465x512: 8, 959x853: 1 | 251 tif + 58 png; 297 RGB, 12 grey |

Key observations
- **Image/mask sizes differ for 2446 of 3809 pairs** (e.g. apo image 1200x800 vs mask 1152x864). Names match 1:1, nothing missing.
  Host's DLTrack training code resizes image and mask *independently* to a square (256x256), i.e. both cover the same
  field of view and are aligned after a full-frame resize → **resize mask to image size (nearest), never crop**.
  (Discussion 702487.) Mask aspect ratios differ from image aspect ratios, so many training images are anisotropically stretched.
- **Duplicates**: 670 identical pixel groups inside fasc_imgs (and 670 in fasc_masks) — matches discussion 740356.
  78 groups shared between apo_imgs and fasc_imgs; 20 within apo_imgs. → group-aware CV split by pixel hash.
- **Test**: 81 test images contain coloured pixels (overlays / coloured rulers); 120 consecutive pairs have thumbnail
  correlation > 0.8 (video clips of ~5 frames). Mixed .tif/.png — keep exact file stem as `image_id` (check sample_submission).
- No read errors.

Open questions / next
- Inspect test images at full resolution for ruler type per device (1200x800 vs 1088x644 vs 1069x853 vs small ~465x512).
- Confirm `image_id` format against sample_submission.csv.

## 2026-10-12 — Stage 1 scale calibration (scripts/01_scale.py → resources/test_scale.csv)

**Submission format** (sample_submission.csv): `;`-separated, `image_id` *includes the extension*
(`IMG_00001.tif`, `IMG_00252.png`). The two example rows are made-up values (host, discussion 743111).

Test images fall into five screen layouts; all 309 calibrated, every one checked visually (ticks + fitted 1 cm grid).

| device | ids | n | ruler | rule | px/mm |
|---|---|---|---|---|---|
| Siemens Acuson Juniper | 1-35, 196-251 (.tif, 1200x800 black) | 91 | right, x~1146-1156 | long ticks = 1 cm (short ticks 2 or 5 mm depending on depth) | 8.70-15.97 (depth 3-7 cm) |
| Philips Lumify | 252-309 (.png, 1200x800 black) | 58 | far left, x~5-10 | 5 mm ticks up to 5 cm depth, **10 mm at 6 cm** (depth read from the "Tiefe" text) | 10.07-20.15 |
| Telemed (1200x800 grey UI) | 66-105, 146-195 | 90 | left line x~66, ticks right of it | 10 mm ticks, depth always 50 mm | 14.80-14.82 |
| Telemed (1088x644) | 56-65, 106-145 | 50 | x~1-8 | 10 mm ticks, depth 50 mm | 12.61 |
| Cropped legacy images | 36-55 (1069x853, 959x853, ~465x513) | 20 | bottom edge | 1 cm ticks (host: assume 1 cm) | 16.67 / 7.75 |

Checks
- Siemens: depth text x px/mm = 609 px for all depths 4-7 cm (constant image height); 3 / 3.5 cm smaller field.
- Lumify: matches the "Tiefe" read-out for all 6 depths (30-60 mm). IMG_00307 is a 5 cm image whose
  screenshot is cut at ~4 cm, so depth-text/height would be wrong there; the tick spacing (15.4 px/mm) is right.
- Grid fitting (RANSAC + missing-tick penalty) rejects bright-tissue false ticks on the cropped images.
- Horizontal Telemed ticks are 13.4 px/mm vs 14.8 vertical (non-square display); host measured with the vertical
  scale only, so we use the vertical value everywhere.

## 2026-10-13 — v1 baseline run (Kaggle commit, GPU T4 x2, ~57 min, code @31f3ea7)

- Aponeurosis U-Net (resnet34, 512x768): 903 train / 125 val after de-dup, best val Dice **0.843** (epoch 24/30, plateau from ~epoch 16).
- Fascicle U-Net: 1799 train / 247 val (715 duplicates dropped), best val Dice **0.414** (prec 0.30 / rec 0.67 — pos_weight pushes recall).
- Geometry on all 309 test images; NaN filled by median: PA/FL 5 images, MT 2 images.
- Median per device (PA deg / FL mm / MT mm / n fascicles):
  cropped 17.7/66.6/18.2/22 · lumify 21.8/65.2/20.6/15 · siemens 15.1/89.1/21.9/24 · tel1088 10.0/114.5/27.1/17 · telemed 10.1/92.9/23.7/11
- Not submitted yet (user decision). Open concerns: Telemed PA ~10 deg looks low; FL depends on extrapolation limit;
  output upload was slow (prob maps + models + repo copied into /kaggle/working) — keep outputs lean next time.
