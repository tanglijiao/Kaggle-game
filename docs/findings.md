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
