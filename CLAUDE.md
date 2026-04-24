# Orthodontics Pipeline — Triton Migration Rules

## Project Context

This is a 3-stage AI pipeline for orthodontic tooth visualization:
- **Stage1**: Face detection (dlib) → Mouth parsing (BiSeNet) → Tooth contour segmentation (UNet)
- **Stage2**: Contour-to-contour diffusion (60-step DDPM loop)
- **Stage3**: Contour-to-tooth image diffusion (60-step DDPM loop)
- **Restore**: Composite generated teeth back onto original photo

The goal is to migrate from the current subprocess-based serving to **NVIDIA Triton Inference Server with a Python backend**.

---

## Critical Rules

### Never Do These

- **Never add `torch.load()` inside `execute()`, `run()`, `predict()`, or any method called per-request.** Model loading belongs exclusively in `initialize()` or `__init__` of Stage classes.
- **Never hardcode `"cuda"` strings.** Always use `self.device` which is set from `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Never call `subprocess.run()` or `os.system()` to invoke main.py.** The entire purpose of this migration is to eliminate subprocess spawning.
- **Never import `triton_python_backend_utils` outside of `model.py`.** It only exists inside the Triton container.
- **Never move GPU tensors to numpy without calling `.cpu()` first** — this will crash silently.

### Always Do These

- **Always call `model.eval()` and `torch.no_grad()` in `initialize()` and inference methods respectively.**
- **Always accept a `device` parameter in refactored Stage classes** so `model.py` controls placement.
- **Always keep the symlink `triton_model_repository/orthodontics_pipeline/1/libs → Code/`** so Stage imports work without copying files.
- **Always create `triton_model_repository/orthodontics_pipeline/1/.keep`** — Triton requires this file to recognize the version directory.
- **Always test with `curl http://localhost:8000/v2/models/orthodontics_pipeline/ready` before wiring up FastAPI.**

---

## File Ownership — What to Touch

| File | Action | Notes |
|---|---|---|
| `Code/Stage1_ToothSegm.py` | Refactor | Add `device` param, move torch.load to __init__ |
| `Code/Stage2_Mask2Mask.py` | Refactor | Add `device` param, move torch.load to __init__ |
| `Code/Stage3_Mask2Teeth.py` | Refactor | Add `device` param, move torch.load to __init__ |
| `backend/app.py` | Refactor | Replace subprocess with tritonclient.http calls |
| `triton_model_repository/` | Create | Does not exist yet — build from scratch |
| `Dockerfile` | Create | Needs cmake + dlib build deps |
| `Code/main.py` | Do NOT touch | Keep for standalone CLI use |
| `Code/Stage2/Network.py` | Do NOT touch | Diffusion loop stays as-is in Python |
| `Code/Stage3/Network.py` | Do NOT touch | Diffusion loop stays as-is in Python |

---

## Stage Class Refactor Contract

After refactoring, each Stage class must satisfy this interface:

```python
class StageN:
    def __init__(self, device: torch.device, config_path: str, **kwargs):
        # Load ALL weights here. Never in run().
        ...

    @torch.no_grad()
    def run(self, input_data, **params) -> dict:
        # Pure inference. Returns dict with named outputs.
        ...
```

Return dict keys expected by `model.py`:

| Stage | Return Keys |
|---|---|
| Stage1 | `contour_mask`, `crop_info`, `debug_image` |
| Stage2 | `generated_contour` |
| Stage3 | `teeth_image` |

---

## Triton Python Backend Contract

`model.py` must implement exactly this class with these three methods:

```python
class TritonPythonModel:
    def initialize(self, args: dict) -> None: ...
    def execute(self, requests: list) -> list: ...
    def finalize(self) -> None: ...
```

- `initialize` is called **once** at server startup
- `execute` receives a list of `InferenceRequest` objects, returns a list of `InferenceResponse` objects of the **same length**
- `finalize` is called at server shutdown for cleanup

---

## Input / Output Tensor Spec

These names must match **exactly** between `config.pbtxt`, `model.py`, and the FastAPI client:

| Tensor | Direction | dtype | dims | Description |
|---|---|---|---|---|
| `image_bytes` | input | BYTES | [1] | Raw PNG/JPEG bytes |
| `whiteness` | input | FP32 | [1] | 0.0–1.0 tooth whiteness control |
| `alignment` | input | FP32 | [1] | Tooth alignment adjustment |
| `sample_num` | input | INT32 | [1] | Number of diffusion samples |
| `seed` | input | INT32 | [1] | RNG seed (-1 = random) |
| `result_image` | output | BYTES | [1] | Final PNG bytes |
| `stage1_debug` | output | BYTES | [1] | Stage1 debug PNG bytes |

---

## Docker Rules

- Base image: `nvcr.io/nvidia/tritonserver:24.01-py3`
- dlib **must** be installed after `cmake libopenblas-dev liblapack-dev libx11-dev`
- `WORKDIR` inside container must be `/app/Code` so relative config paths (`./Stage2/config/...`) resolve correctly
- Weights directory must be mounted or copied to match paths in `Config.yaml`

---

## Port Assignments

| Port | Protocol | Use |
|---|---|---|
| 8000 | HTTP/REST | Triton inference + health checks |
| 8001 | gRPC | Triton gRPC inference |
| 8002 | HTTP | Triton metrics (Prometheus) |
| 8080 | HTTP | FastAPI gateway |

---

## Debugging Workflow

When something breaks, check in this order:

1. **Triton server logs**: `docker logs <container> 2>&1 | grep -E "ERROR|WARN|orthodontics"`
2. **Model ready**: `curl http://localhost:8000/v2/models/orthodontics_pipeline/ready`
3. **Python traceback**: look for `TritonModelException` in server logs
4. **Tensor shape mismatch**: check `dims` in config.pbtxt matches actual numpy array shapes
5. **Import error**: confirm `sys.path.insert(0, CODE_DIR)` is first line after standard imports in model.py

---

## Out of Scope for This Migration

The following are **not** part of the Triton Python backend migration and should be addressed separately:

- ONNX export of Stage2/Stage3 denoiser (diffusion loop is Python-controlled, not a single graph)
- TensorRT optimization
- Multi-GPU scaling
- Authentication / SSL on Triton endpoints
- Model versioning beyond version `1`