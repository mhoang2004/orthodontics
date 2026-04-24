# DONE — Stage1 Refactoring

**Date:** 2026-04-24
**Contract:** Stage-Class-Refactor-Contract (CLAUDE.md)

---

## Summary

Refactored Stage1 (tooth segmentation pipeline) from a single function into a
class-based architecture ready for NVIDIA Triton Python backend integration.

---

## Files Modified

| File | Change |
|---|---|
| `Code/Stage1_ToothSegm.py` | New `Stage1Class` + backward-compat `Stage1()` wrapper |
| `Code/Stage1/DetectFace/DetectFace.py` | Added `load_face_models()`, DI params on `face_landmark_detect` & `DetectFace` |
| `Code/Stage1/DetectMouth/test.py` | Added `load_bisenet()`, DI params on `evaluate()` |
| `Code/Stage1/DetectMouth/DetectMouth.py` | DI params (`bisenet`, `device`) on `DetectMouth()` |
| `Code/Stage1/SegmentToothContour/SegmentToothContour.py` | Added `load_unet()`, DI params + `self.device` usage |

## Files NOT Modified

| File | Reason |
|---|---|
| `Code/Stage2_Mask2Mask.py` | Out of scope |
| `Code/Stage3_Mask2Teeth.py` | Out of scope |
| `Code/main.py` | Legacy CLI — no touch |
| `Code/Stage1/SegmentTeeth/DetectContour.py` | Pure OpenCV, no models — no change needed |

---

## Contract Compliance

### Class Signature

```python
class Stage1Class:
    def __init__(self, device: torch.device, config_path: str = None, *,
                 mode: str = 'C2C2T_v2', state: str = None,
                 bisenet_cp: str = '...', dlib_weight_path: str = '...'):
        # ALL model loading here — never in run()
        ...

    @torch.no_grad()
    def run(self, input_data, *, if_visual: bool = False, **params) -> dict:
        # Pure inference only
        ...
```

### Return Keys

| Key | Contract | Backward-compat |
|---|---|---|
| `contour_mask` | ✅ | — |
| `crop_info` | ✅ | — |
| `debug_image` | ✅ | — |
| `ori_face` | — | ✅ |
| `detect_face` | — | ✅ |
| `info` | — | ✅ |
| `crop_face` | — | ✅ |
| `crop_mouth` | — | ✅ |
| `crop_teeth` | — | ✅ |
| `crop_mask` | — | ✅ |

---

## Device Placement Rules

| Model | Device | Rationale |
|---|---|---|
| dlib `get_frontal_face_detector()` | **CPU** | dlib has no CUDA support |
| dlib `shape_predictor` | **CPU** | dlib has no CUDA support |
| BiSeNet (mouth parsing) | `self.device` | GPU-capable PyTorch model |
| UNet (tooth contour) | `self.device` | GPU-capable PyTorch model |

---

## Backward Compatibility

The legacy function `Stage1(img_path, mode, state, if_visual=False)` is
preserved as a thin wrapper.  All existing call-sites (`main.py`,
`main_ForFolder.py`, `main_interactive.py`) continue to work without
modification:

```python
from Stage1_ToothSegm import Stage1
stage1_data = Stage1(img_path, mode=args.mode, state=args.stage1, if_visual=False)
```

### Triton Usage (new)

```python
from Stage1_ToothSegm import Stage1Class

device = torch.device("cuda")
stage1 = Stage1Class(device=device, mode="C2C2T_v2", state="path/to/unet.pth")
# Models loaded once ↑

result = stage1.run("path/to/image.jpg")  # Pure inference ↓
```

---

## Key Design Decisions

1. **Dependency Injection pattern** — All sub-modules (`DetectFace`, `DetectMouth`,
   `SegmentToothContour`) accept optional pre-loaded models. When `None`, they
   fall back to per-call loading for standalone use.

2. **No hardcoded `"cuda"`** — All GPU placement uses `self.device` or the
   injected `device` parameter. Eliminated bare `.cuda()` calls in
   `SegmentToothContour.py`.

3. **`map_location=device`** — All `torch.load()` calls use `map_location` to
   load directly onto the target device, avoiding CPU→GPU copies.

4. **Class named `Stage1Class`** — The function `Stage1()` must keep its name
   for backward compatibility (`from Stage1_ToothSegm import Stage1`). The
   class uses `Stage1Class` to avoid Python name-shadowing issues.
