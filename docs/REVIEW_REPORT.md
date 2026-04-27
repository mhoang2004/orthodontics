# Architecture Review Report — All Stages

**Date:** 2026-04-24  
**Scope:** Stage1, Stage2, Stage3 refactored files + downstream Generator/Network modules  
**Reviewer:** Antigravity (automated architecture review)  
**Verdict:** ⛔ **Not production-ready** — 4 critical issues must be resolved before Triton deployment

---

## Files Reviewed

| File | Lines | Role |
|---|---|---|
| [Stage1_ToothSegm.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1_ToothSegm.py) | 157 | Face → mouth → tooth contour |
| [Stage2_Mask2Mask.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2_Mask2Mask.py) | 145 | Contour-to-contour diffusion |
| [Stage3_Mask2Teeth.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3_Mask2Teeth.py) | 213 | Contour-to-teeth image diffusion |
| [Stage2/Generator.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Generator.py) | 233 | Stage2 Generator classes |
| [Stage3/Generator.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py) | 677 | Stage3 Generator classes |
| [Stage2/Network.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Network.py) | 897 | Diffusion network + noise schedule |
| [Stage1/DetectFace/DetectFace.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/DetectFace/DetectFace.py) | 90 | dlib face detection |
| [Stage1/DetectMouth/DetectMouth.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/DetectMouth/DetectMouth.py) | 97 | BiSeNet mouth parsing |
| [Stage1/DetectMouth/test.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/DetectMouth/test.py) | 119 | BiSeNet loader + evaluate |
| [Stage1/SegmentToothContour/SegmentToothContour.py](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/SegmentToothContour/SegmentToothContour.py) | 74 | UNet tooth contour segmenter |

---

## 🔴 Critical Issues (Production Blockers)

### C1 — Hardcoded `.cuda()` in ALL Generator classes destroys device discipline

> [!CAUTION]
> This is the **single most dangerous issue** in the entire codebase. The refactored Stage classes correctly use `self.device`, but every Generator they instantiate immediately calls `.cuda()`, bypassing the device abstraction entirely.

**Files affected:**
- [Stage2/Generator.py:12](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Generator.py#L12) — `Mask2MaskGenerator.__init__`: `self.netG = network.cuda()`
- [Stage2/Generator.py:25-27](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Generator.py#L25-L27) — `set_input()`: `.cuda()` on every tensor
- [Stage2/Generator.py:133](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Generator.py#L133) — `Contour2ContourGenerator.__init__`: `self.netG = network.cuda()`
- [Stage2/Generator.py:143-146](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Generator.py#L143-L146) — `set_input()`: `.cuda()`
- [Stage3/Generator.py:16](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L16) — `Mask2TeethGenerator.__init__`: `.cuda()`
- [Stage3/Generator.py:29-32](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L29-L32) — `set_input()`: `.cuda()`
- [Stage3/Generator.py:243](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L243) — `Contour2TeethGenerator.__init__`: `.cuda()`
- [Stage3/Generator.py:253-256](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L253-L256) — `set_input()`: `.cuda()`
- [Stage3/Generator.py:325](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L325) — `Contour2ToothGenerator_FaceColor_TeethColor.__init__`: `.cuda()`
- [Stage3/Generator.py:335-338](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L335-L338) — `set_input()`: `.cuda()`
- [Stage3/Generator.py:418](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L418) — `Contour2ToothGenerator_FaceColor_LightColor.__init__`: `.cuda()`
- [Stage3/Generator.py:576](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L576) — `Contour2ToothGenerator_Fourier.__init__`: `.cuda()`

**Impact:** 
- **CPU-only environments crash immediately** — any machine without CUDA gets `RuntimeError: CUDA not available`.
- **Triton with multi-GPU**: device assignment is ignored; all models pile onto `cuda:0`.
- The careful `self.device` plumbing in Stage2/Stage3 `__init__` (`self.netG.to(self.device)`) is then **overridden** when the Generator calls `network.cuda()` in its own `__init__`, potentially moving the model back to `cuda:0` even if `self.device` was `cuda:1`.
- The `set_input()` methods also use bare `.cuda()` instead of `.to(device)`.

**Also in Network.py:**
- [Stage2/Network.py:723](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Network.py#L723) — `set_new_noise_schedule(self, device=torch.device('cuda'))` — hardcoded default `cuda` device. When a Generator calls `self.netG.set_new_noise_schedule()` without passing a device argument, the noise schedule buffers land on `cuda:0` regardless of where the model parameters are.

---

### C2 — Missing `.cpu()` in Stage3's `tensor2img` — will crash on GPU tensors

> [!CAUTION]
> CLAUDE.md rule: "Never move GPU tensors to numpy without calling `.cpu()` first — this will crash silently."

**Location:** [Stage3_Mask2Teeth.py:19](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3_Mask2Teeth.py#L19)

```python
# Stage3 tensor2img — NO .cpu() call:
tensor = tensor.clamp_(*min_max)  # stays on GPU
img_np = tensor.numpy()           # 💥 RuntimeError: can't convert CUDA tensor to numpy

# Stage2 tensor2img — CORRECT:
tensor = tensor.cpu().clamp_(*min_max)  # moved to CPU first ✅
```

Stage2's `tensor2img` was correctly patched to call `.cpu()` first (line 19 of Stage2_Mask2Mask.py). Stage3's **identical** function was **not** patched.

**Why it doesn't crash today:** The `run()` method on line 175 calls `tensor2img(prediction.cpu())`, so the tensor arrives on CPU. But:
1. The `visual_imgs` path in Stage2 (line 118) calls `tensor2img(v)` on tensors that might still be on GPU — same vulnerability.
2. If anyone calls `tensor2img` directly on a GPU tensor, it silently crashes.
3. The in-place `clamp_` modifies the original tensor, potentially corrupting upstream data.

---

### C3 — Unhandled face detection failure in Stage1 crashes the entire pipeline

**Location:** [DetectFace.py:40-43](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/DetectFace/DetectFace.py#L40-L43)

```python
def face_landmark_detect(...):
    faces = face_detector(img, 0)
    try:
        shape = shape_predictor(img, faces[0])
    except:
        return None, None        # ← returns None
```

Then in [DetectFace.py:63-78](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/DetectFace/DetectFace.py#L63-L78):
```python
def DetectFace(img_path, ...):
    face, landmarks = face_landmark_detect(...)
    # face is None here if no face detected
    y1 = face.top()  # 💥 AttributeError: 'NoneType' object has no attribute 'top'
```

**Impact:** Any image without a detectable face causes an unrecoverable crash. In a Triton server this kills the entire inference request with no meaningful error message. The bare `except:` clause also swallows all exceptions (including `KeyboardInterrupt`, `SystemExit`, OOM errors).

---

### C4 — Diffusion visual tensor accumulation — unbounded GPU memory growth

**Location:** [Network.py:789-806](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Network.py#L789-L806) (identical in Stage3/Network.py)

```python
def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
    ret_arr = y_t                                    # starts with initial noise
    for i in reversed(range(0, self.num_timesteps)):  # 60 iterations
        ...
        if i % sample_inter == 0:
            ret_arr = torch.cat([ret_arr, y_t], dim=0)  # grows every sample_inter steps
    return y_t, ret_arr
```

With `num_timesteps=60` and `sample_num=5`, `sample_inter=12`, this creates `ret_arr` of shape `[6, 3, H, W]` — manageable. But:
1. The entire `ret_arr` lives **on GPU** during the loop.
2. If `sample_num` is increased (e.g., the interactive API allows user control), `sample_inter` decreases and `ret_arr` grows proportionally.
3. With `sample_num=60`, every step is captured → `ret_arr` = `[61, 3, H, W]` on GPU.
4. Additionally, in Generator's `predict_with_visuals`, each visual step is **copied again** via numpy round-trip (`.cpu().numpy()` then `torch.from_numpy()`), doubling memory briefly.

**Impact:** Under concurrent Triton requests, GPU OOM is likely with large `sample_num` values.

---

## 🟡 Medium Issues

### M1 — Legacy wrappers reload ALL models on every call

**Location:** 
- [Stage1_ToothSegm.py:141-150](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1_ToothSegm.py#L141-L150)
- [Stage2_Mask2Mask.py:135-144](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2_Mask2Mask.py#L135-L144)
- [Stage3_Mask2Teeth.py:193-212](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3_Mask2Teeth.py#L193-L212)

All three legacy wrappers instantiate the Stage class (loading all weights) on every single call. While documented as deprecated, they are the **only** entry points used by `main.py`. Each call to the legacy `Stage2_Mask2Mask()` function loads ~100MB+ of diffusion weights from disk, moves them to GPU, builds noise schedule, then discards everything after inference.

**Risk:** If `main.py` is used in any loop or batch scenario, this is catastrophically slow and fragments GPU memory.

---

### M2 — `result_vis/` directory assumed to exist for `if_visual=True`

**Locations:**
- [DetectMouth.py:81-82](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/DetectMouth/DetectMouth.py#L81-L82) — `cv2.imwrite('./result_vis/...')` — **no `os.makedirs`**
- [SegmentToothContour.py:72](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/SegmentToothContour/SegmentToothContour.py#L72) — `cv2.imwrite('./result_vis/...')` — **no `os.makedirs`**
- [Stage3_Mask2Teeth.py:178](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3_Mask2Teeth.py#L178) — `cv2.imwrite('./result_vis/...')` — **no `os.makedirs`**

Stage2 correctly calls `os.makedirs('./result_vis', exist_ok=True)` on line 121, but the other stages do not. `cv2.imwrite` silently fails on a non-existent directory (returns `False` without raising), so debug images are silently lost.

---

### M3 — Generator classes store mutable state on `self` — not thread-safe

**Locations:** All Generator `set_input()` + `predict()` methods:
```python
def set_input(self, data):
    self.bf_image = data.get('bf_image').cuda()
    self.cond_image = data.get('cond_image').cuda()
    ...

def predict(self, data):
    ...
    self.set_input({...})
    self.output, self.visuals = self.netG.restoration(...)
```

`self.bf_image`, `self.cond_image`, `self.output`, `self.visuals` are **per-request state stored on the class instance**. If Triton dispatches two concurrent requests to the same model instance (which it does by default), these attributes will be overwritten by the second request while the first is still using them.

**Impact:** Data corruption / wrong results under concurrent Triton requests. This is a **race condition**.

---

### M4 — `config_path` parameter semantics inconsistent across stages

| Stage | `config_path` means | Contract says |
|---|---|---|
| Stage1 | Unused (ignored) | Config YAML path |
| Stage2 | Config YAML path (fallback, can be `None`) | Config YAML path |
| Stage3 | **Checkpoint `.pth` path** (model weights) | Config YAML path |

Stage3's `__init__` [line 85-89](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3_Mask2Teeth.py#L85-L89) uses `config_path` to load model weights via `torch.load(config_path, ...)`. The contract in CLAUDE.md says `config_path: str` should be a config path. This naming mismatch will confuse anyone writing `model.py`.

---

### M5 — `set_new_noise_schedule()` called with default `device='cuda'` inside Generators

The Generators call `self.netG.set_new_noise_schedule()` without passing a device argument:
- [Stage2/Generator.py:13](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Generator.py#L13)
- [Stage2/Generator.py:134](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Generator.py#L134)
- [Stage3/Generator.py:17](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3/Generator.py#L17)
- etc.

[Network.py:723](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2/Network.py#L723): `def set_new_noise_schedule(self, device=torch.device('cuda')):`

The noise schedule buffers (`gammas`, `sqrt_recip_gammas`, etc.) are created on `cuda:0` regardless of where the model actually lives. This causes **device mismatch** errors if the model is on `cpu` or `cuda:1`.

> [!WARNING]
> This compounds with C1. Even if you fix the Generator `.cuda()` calls, the noise schedule buffers will still be on the wrong device unless `set_new_noise_schedule` is also updated.

---

## 🟢 Low Priority Improvements

### L1 — Duplicated `tensor2img` function

`tensor2img()` is copy-pasted identically in both [Stage2_Mask2Mask.py:13-35](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2_Mask2Mask.py#L13-L35) and [Stage3_Mask2Teeth.py:13-35](file:///home/trislord/Dev/Python/orthodontics/Code/Stage3_Mask2Teeth.py#L13-L35) (with the `.cpu()` bug in Stage3's copy). Should be extracted to a shared utility module.

---

### L2 — `_MODE_REGISTRY` in Stage3 uses dynamic import but Stage2 uses inline `from ... import`

Stage3 uses `importlib.import_module()` with a registry dict — clean and extensible. Stage2 uses inline `from Stage2.Generator import ...` inside `__init__`. Should be unified for consistency.

---

### L3 — Bare `except:` clause in face detection

[DetectFace.py:42](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1/DetectFace/DetectFace.py#L42): `except:` catches **everything**, including `SystemExit`, `KeyboardInterrupt`, and `MemoryError`. Should be `except (IndexError, Exception):` at minimum.

---

### L4 — `json` import unused in Stage2_Mask2Mask.py

[Stage2_Mask2Mask.py:2](file:///home/trislord/Dev/Python/orthodontics/Code/Stage2_Mask2Mask.py#L2): `import json` — never used. Minor dead code.

---

### L5 — Stage3 `_MODE_REGISTRY` fragile to class renames

The registry maps mode strings to `(module_path, class_name, yaml_path)` tuples. Class names are strings resolved at runtime via `importlib` — any rename in `Stage3/Generator.py` will cause a silent `AttributeError` at init, not at import time. No static analysis can catch this.

---

### L6 — Windows-style path in `__main__` block

[Stage1_ToothSegm.py:154](file:///home/trislord/Dev/Python/orthodontics/Code/Stage1_ToothSegm.py#L154): `r'C:\IDEA_Lab\...'` — hardcoded Windows path in a Linux/Docker-targeted project. Harmless but confusing.

---

## ⛔ Production Blockers Checklist

| # | Issue | Severity | Blocks Triton? | Blocks CPU? |
|---|---|---|---|---|
| C1 | Hardcoded `.cuda()` in Generators | 🔴 Critical | ✅ Multi-GPU broken | ✅ Crashes |
| C2 | Missing `.cpu()` in Stage3 `tensor2img` | 🔴 Critical | ⚠️ Latent | ✅ If called directly |
| C3 | Unhandled face detection `None` | 🔴 Critical | ✅ Server crash | ✅ CLI crash |
| C4 | Unbounded `ret_arr` GPU accumulation | 🔴 Critical | ✅ OOM risk | ⚠️ RAM risk |
| M3 | Thread-unsafe Generator state | 🟡 Medium | ✅ Race condition | ❌ |
| M5 | Noise schedule on wrong device | 🟡 Medium | ✅ Device mismatch | ✅ Crashes on CPU |

---

## Recommended Fix Priority

```
1. Fix C1 + M5 together  →  Propagate device through Generator + Network
2. Fix C2               →  Add .cpu() to Stage3 tensor2img  
3. Fix C3               →  Graceful error on no-face-detected
4. Fix C4               →  Cap sample_num, move ret_arr to CPU incrementally
5. Fix M3               →  Make Generator methods functional (pass data, don't store on self)
```

> [!IMPORTANT]
> Issues C1 and M5 are **systemic** — they exist in files marked "Do NOT touch" in CLAUDE.md (`Stage2/Network.py`, `Stage3/Network.py`). The contract must be updated to allow modifying Generator and Network files for device discipline, or a wrapper/adapter layer must be introduced.
