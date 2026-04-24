---
name: triton-orthodontics-deployment
description: >
  Deploy the orthodontics 3-stage AI pipeline (Stage1 tooth segmentation,
  Stage2 contour diffusion, Stage3 teeth generation) onto NVIDIA Triton
  Inference Server using a Python backend. Use this skill whenever the user
  asks to: set up Triton, migrate from subprocess to Triton, refactor Stage
  classes for warm loading, write config.pbtxt, write model.py for Triton
  Python backend, create the Dockerfile for Triton + dlib, or wire FastAPI
  to call Triton instead of spawning subprocesses. Also trigger for any
  question about model warm loading, reducing per-request latency, or
  serving .pth PyTorch models through Triton.
---

# Triton Deployment — Orthodontics Pipeline

## Project Layout (read-only reference)

```
orthodontics/
├── backend/app.py              ← FastAPI gateway (currently uses subprocess)
├── Code/
│   ├── main.py                 ← top-level orchestrator
│   ├── main_interactive.py
│   ├── requirements.txt
│   ├── Config.yaml
│   ├── Stage1_ToothSegm.py     ← Stage1 wrapper (loads model per call — BAD)
│   ├── Stage2_Mask2Mask.py     ← Stage2 wrapper (loads model per call — BAD)
│   ├── Stage3_Mask2Teeth.py    ← Stage3 wrapper (loads model per call — BAD)
│   ├── Stage1/
│   │   ├── DetectFace/         ← dlib face detector + landmarks
│   │   ├── DetectMouth/        ← BiSeNet mouth parser
│   │   ├── SegmentToothContour/← UNet contour segmentor
│   │   └── SegmentTeeth/
│   ├── Stage2/
│   │   ├── Network.py          ← diffusion model, 60-step p_sample loop
│   │   ├── Generator.py
│   │   └── config/config_Contour2Contour.yaml
│   ├── Stage3/
│   │   ├── Network.py          ← diffusion model, 60-step p_sample loop
│   │   ├── Generator.py
│   │   └── config/config_Contour2Tooth_facecolor_lightcolor.yaml
│   └── Restore/Restore.py
└── triton_model_repository/    ← CREATE THIS (does not exist yet)
```

---

## Core Problems to Fix

1. **Subprocess anti-pattern**: `backend/app.py` spawns `python main.py` per request → model reloaded from `.pth` every call.
2. **No warm model**: Stage1/2/3 wrapper classes instantiate and load weights inside each call.
3. **No Triton model repository**: needs to be created from scratch.

---

## Target Architecture

```
FastAPI (port 8080)
    │  tritonclient.http
    ▼
Triton Server (port 8000/8001/8002)
    │  Python backend
    ▼
orthodontics_pipeline/1/model.py
    ├── initialize()  ← load ALL weights ONCE at startup
    └── execute()     ← inference only, no torch.load()
          ├── Stage1: dlib + BiSeNet + UNet
          ├── Stage2: diffusion 60-step loop
          ├── Stage3: diffusion 60-step loop
          └── Restore: paste back to original image
```

---

## Step 1 — Create Triton Model Repository Structure

```bash
mkdir -p triton_model_repository/orthodontics_pipeline/1
touch triton_model_repository/orthodontics_pipeline/1/.keep

# Symlink Code/ so model.py can import existing modules without copying
ln -s $(pwd)/Code triton_model_repository/orthodontics_pipeline/1/libs
```

---

## Step 2 — Write config.pbtxt

File: `triton_model_repository/orthodontics_pipeline/config.pbtxt`

```protobuf
name: "orthodontics_pipeline"
backend: "python"
max_batch_size: 1

input [
  { name: "image_bytes"  data_type: TYPE_BYTES  dims: [1] },
  { name: "whiteness"    data_type: TYPE_FP32   dims: [1] },
  { name: "alignment"    data_type: TYPE_FP32   dims: [1] },
  { name: "sample_num"   data_type: TYPE_INT32  dims: [1] },
  { name: "seed"         data_type: TYPE_INT32  dims: [1] }
]

output [
  { name: "result_image"  data_type: TYPE_BYTES  dims: [1] },
  { name: "stage1_debug"  data_type: TYPE_BYTES  dims: [1] }
]

instance_group [{ kind: KIND_GPU, count: 1 }]
```

**If no GPU is available** (dev/demo mode), replace instance_group with:
```protobuf
instance_group [{ kind: KIND_CPU, count: 1 }]
```

---

## Step 3 — Refactor Stage Classes (CRITICAL)

Each Stage*.py must be refactored so that:
- `__init__` loads the model ONCE and stores it on `self`
- Inference methods only call forward pass, never `torch.load()`

### Pattern to apply to Stage2_Mask2Mask.py and Stage3_Mask2Teeth.py

```python
class Stage2:
    def __init__(self, device, config_path, weights_path):
        import yaml
        from Stage2.Generator import Generator

        self.device = device
        with open(config_path) as f:
            self.opt = yaml.safe_load(f)

        # Override weight path from config if needed
        self.model = Generator(self.opt)
        self.model.load_network(weights_path)
        self.model.netG.to(self.device).eval()

    @torch.no_grad()
    def run(self, contour_mask, alignment=0.0, sample_num=1):
        return self.model.predict_interactive(
            contour_mask, alignment=alignment, sample_num=sample_num
        )
```

Apply the same pattern to Stage1 (dlib detector, BiSeNet, UNet each stored as `self.xxx`).

### Key checklist for each Stage class:
- [ ] No `torch.load()` inside `run()` / `predict()` / `__call__()`
- [ ] Model moved to correct device in `__init__`
- [ ] `model.eval()` called in `__init__`
- [ ] `@torch.no_grad()` on inference methods
- [ ] No hardcoded `cuda` strings — use `self.device`

---

## Step 4 — Write model.py (Triton Python Backend)

File: `triton_model_repository/orthodontics_pipeline/1/model.py`

### Skeleton

```python
import triton_python_backend_utils as pb_utils
import numpy as np, torch, sys, os, io, json, logging
from PIL import Image

CODE_DIR = os.path.join(os.path.dirname(__file__), "libs")
sys.path.insert(0, CODE_DIR)

logger = logging.getLogger("orthodontics")

class TritonPythonModel:

    def initialize(self, args):
        """Called ONCE at server startup. Load all models here."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        cfg = self._read_config(args)

        # Load each stage — pass device explicitly
        from Stage1_ToothSegm import Stage1
        from Stage2_Mask2Mask import Stage2
        from Stage3_Mask2Teeth import Stage3

        self.stage1 = Stage1(device=self.device, **cfg["stage1"])
        self.stage2 = Stage2(device=self.device, **cfg["stage2"])
        self.stage3 = Stage3(device=self.device, **cfg["stage3"])
        logger.info("All stages loaded on %s", self.device)

    def execute(self, requests):
        return [self._run(r) for r in requests]

    def _run(self, request):
        def get(name, dtype):
            return pb_utils.get_input_tensor_by_name(request, name).as_numpy()

        img_bytes  = bytes(get("image_bytes",  object)[0])
        whiteness  = float(get("whiteness",  np.float32)[0])
        alignment  = float(get("alignment",  np.float32)[0])
        sample_num = int(get("sample_num",   np.int32)[0])
        seed       = int(get("seed",         np.int32)[0])

        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        s1 = self.stage1.run(image)
        s2 = self.stage2.run(s1["contour_mask"], alignment=alignment)
        s3 = self.stage3.run(
            s2["generated_contour"],
            whiteness=whiteness,
            sample_num=sample_num
        )

        from Restore.Restore import restore
        final = restore(
            original=image,
            teeth=s3["teeth_image"],
            crop_info=s1["crop_info"]
        )

        def encode(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("result_image",
                np.array([encode(final)],             dtype=object)),
            pb_utils.Tensor("stage1_debug",
                np.array([encode(s1["debug_image"])], dtype=object)),
        ])

    def finalize(self):
        del self.stage1, self.stage2, self.stage3
        torch.cuda.empty_cache()

    def _read_config(self, args):
        """Return paths for each stage from model_config parameters or defaults."""
        # Implement based on actual weight paths in the project
        return {
            "stage1": {},  # fill with actual paths
            "stage2": {"config_path": "Stage2/config/config_Contour2Contour.yaml"},
            "stage3": {"config_path": "Stage3/config/config_Contour2Tooth_facecolor_lightcolor.yaml"},
        }
```

---

## Step 5 — Refactor FastAPI (backend/app.py)

Replace every `subprocess.run(["python", "main.py", ...])` call with:

```python
import tritonclient.http as httpclient
import numpy as np

_triton = httpclient.InferenceServerClient(url="localhost:8000")

def call_pipeline(image_bytes, whiteness, alignment, sample_num, seed=-1):
    inputs = [
        httpclient.InferInput("image_bytes", [1], "BYTES"),
        httpclient.InferInput("whiteness",   [1], "FP32"),
        httpclient.InferInput("alignment",   [1], "FP32"),
        httpclient.InferInput("sample_num",  [1], "INT32"),
        httpclient.InferInput("seed",        [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(np.array([image_bytes], dtype=object))
    inputs[1].set_data_from_numpy(np.array([whiteness],   dtype=np.float32))
    inputs[2].set_data_from_numpy(np.array([alignment],   dtype=np.float32))
    inputs[3].set_data_from_numpy(np.array([sample_num],  dtype=np.int32))
    inputs[4].set_data_from_numpy(np.array([seed],        dtype=np.int32))

    outputs = [
        httpclient.InferRequestedOutput("result_image"),
        httpclient.InferRequestedOutput("stage1_debug"),
    ]
    result = _triton.infer("orthodontics_pipeline", inputs, outputs=outputs)
    return (
        bytes(result.as_numpy("result_image")[0]),
        bytes(result.as_numpy("stage1_debug")[0]),
    )
```

---

## Step 6 — Dockerfile

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# dlib build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake libopenblas-dev liblapack-dev libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps from project
COPY Code/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# dlib must be built from source inside container
RUN pip install --no-cache-dir dlib

# Weights and model repository
COPY Code /app/Code
COPY triton_model_repository /models

WORKDIR /app/Code
EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
```

**For CPU-only demo** (no GPU required):
```bash
docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  orthodontics-triton
```

**For GPU**:
```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  orthodontics-triton
```

---

## Health Check

```bash
# Server alive?
curl http://localhost:8000/v2/health/live

# Model ready?
curl http://localhost:8000/v2/models/orthodontics_pipeline/ready

# List loaded models
curl http://localhost:8000/v2/models
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: dlib` | dlib not in container | Add `pip install dlib` after cmake deps in Dockerfile |
| `CUDA error: no kernel image` | CUDA version mismatch | Match PyTorch CUDA version to Triton base image |
| `model not found` | Missing `.keep` file in `1/` folder | `touch triton_model_repository/orthodontics_pipeline/1/.keep` |
| `TypeError: can't convert CUDA tensor to numpy` | Tensor still on GPU when encoding | Call `.cpu().numpy()` before passing to pb_utils |
| `triton_python_backend_utils not found` | Running model.py outside Triton | Only import this inside Triton container |
| Slow first request | Model lazy-loaded | Ensure all `torch.load()` is in `initialize()`, not `execute()` |

---

## Incremental Optimization Path (after demo works)

1. **Now**: Python backend — full pipeline in one model.py
2. **Next**: Export Stage2/Stage3 denoiser U-Net to ONNX (single forward pass only, not the loop)
3. **Later**: Replace Python denoiser call with Triton sub-model call inside the loop
4. **Advanced**: TensorRT conversion of denoiser for maximum GPU throughput