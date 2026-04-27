from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import os
import aiofiles
import io
import base64
import numpy as np

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import InferenceServerException
except Exception:
    httpclient = None
    InferenceServerException = Exception

app = FastAPI()

# 1. Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Thiết lập đường dẫn hệ thống
# Lấy thư mục gốc dựa trên vị trí file này (giả sử file này nằm trong subfolder của project)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(ROOT, "Code")
OUTPUT_DIR = os.path.join(ROOT, "Output")

MAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prediction")
PROCESS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "processing")

# Đảm bảo các thư mục tồn tại
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESS_OUTPUT_DIR, exist_ok=True)

# Cấu hình Timeout (mặc định 30 phút)
PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT", "1800"))
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "dental_pipeline")

DEFAULT_WHITENESS = float(os.getenv("DEFAULT_WHITENESS", "1.0"))
DEFAULT_ALIGNMENT = float(os.getenv("DEFAULT_ALIGNMENT", "1.0"))
DEFAULT_SAMPLE_NUM = int(os.getenv("DEFAULT_SAMPLE_NUM", "60"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "-1"))

_triton_client = None


def _get_triton_client():
    global _triton_client
    if _triton_client is None:
        if httpclient is None:
            raise RuntimeError("Không tìm thấy thư viện tritonclient")
        _triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
    return _triton_client


def _as_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _call_pipeline_triton(
    image_bytes: bytes,
    whiteness: float,
    alignment: float,
    sample_num: int,
    seed: int,
):
    client = _get_triton_client()

    inputs = [
        httpclient.InferInput("IMAGE", [1], "BYTES"),
        httpclient.InferInput("whiteness", [1], "FP32"),
        httpclient.InferInput("alignment", [1], "FP32"),
        httpclient.InferInput("sample_num", [1], "INT32"),
        httpclient.InferInput("seed", [1], "INT32"),
    ]

    inputs[0].set_data_from_numpy(np.array([image_bytes], dtype=object))
    inputs[1].set_data_from_numpy(np.array([whiteness], dtype=np.float32))
    inputs[2].set_data_from_numpy(np.array([alignment], dtype=np.float32))
    inputs[3].set_data_from_numpy(np.array([sample_num], dtype=np.int32))
    inputs[4].set_data_from_numpy(np.array([seed], dtype=np.int32))

    outputs = [
        httpclient.InferRequestedOutput("PRED_IMAGE"),
        httpclient.InferRequestedOutput("METADATA"),
    ]

    result = client.infer(TRITON_MODEL_NAME, inputs, outputs=outputs)

    result_image = result.as_numpy("PRED_IMAGE")
    stage1_debug = result.as_numpy("METADATA")

    if result_image is None or len(result_image) == 0:
        raise RuntimeError("Không nhận được result_image từ Triton")

    result_image_bytes = _as_bytes(result_image[0])
    stage1_debug_bytes = b""
    if stage1_debug is not None and len(stage1_debug) > 0:
        stage1_debug_bytes = _as_bytes(stage1_debug[0])

    return result_image_bytes, stage1_debug_bytes


async def _call_pipeline_triton_with_timeout(
    image_bytes: bytes,
    whiteness: float,
    alignment: float,
    sample_num: int,
    seed: int,
):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _call_pipeline_triton,
                image_bytes,
                whiteness,
                alignment,
                sample_num,
                seed,
            ),
            timeout=PIPELINE_TIMEOUT,
        )
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"Hết thời gian xử lý ({PIPELINE_TIMEOUT}s)") from e

@app.get("/")
async def root():
    return {"status": "ok", "message": "Orthodontics API is running"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # 3. Xử lý tên file và định dạng
    uid = str(uuid.uuid4())
    
    # Lấy extension gốc của file khách gửi (.jpg, .png, .jpeg, ...)
    original_ext = os.path.splitext(file.filename)[1].lower()
    if original_ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ định dạng JPG hoặc PNG")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File rỗng")

        print(f"--- Đang xử lý ảnh bằng Triton: {file.filename} ---")
        file_bytes, _ = await _call_pipeline_triton_with_timeout(
            image_bytes=content,
            whiteness=DEFAULT_WHITENESS,
            alignment=DEFAULT_ALIGNMENT,
            sample_num=DEFAULT_SAMPLE_NUM,
            seed=DEFAULT_SEED,
        )
        
        return StreamingResponse(
            io.BytesIO(file_bytes), 
            media_type="image/png", 
            headers={"Content-Disposition": f"attachment; filename=result_{uid}.png"}
        )

    except TimeoutError:
        raise HTTPException(status_code=504, detail=f"Hết thời gian xử lý ({PIPELINE_TIMEOUT}s)")
    except InferenceServerException as e:
        print(f"Triton Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)[:500]}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"System Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


@app.post("/infer-interactive")
async def infer_interactive(
    file: UploadFile = File(...),
    whiteness: float = Form(1.0),
    alignment: float = Form(1.0),
    timesteps: int = Form(60),
):
    """
    Interactive inference endpoint.
    
    Tham số:
    - file: Ảnh đầu vào (JPG/PNG)
    - whiteness: Độ trắng răng (0.0 - 2.0, mặc định 1.0)
    - alignment: Cường độ alignment contour (0.0 - 2.0, mặc định 1.0) 
    - timesteps: Số bước diffusion Stage 3 (10 - 200, mặc định 60)
    """
    uid = str(uuid.uuid4())
    
    original_ext = os.path.splitext(file.filename)[1].lower()
    if original_ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ định dạng JPG hoặc PNG")

    # Clamp parameters
    whiteness = max(0.0, min(2.0, whiteness))
    alignment = max(0.0, min(2.0, alignment))
    timesteps = max(10, min(200, timesteps))

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File rỗng")

        print(
            f"--- Interactive infer (Triton): "
            f"whiteness={whiteness}, alignment={alignment}, timesteps={timesteps} ---"
        )

        file_bytes, _ = await _call_pipeline_triton_with_timeout(
            image_bytes=content,
            whiteness=whiteness,
            alignment=alignment,
            sample_num=timesteps,
            seed=DEFAULT_SEED,
        )

        img_b64 = base64.b64encode(file_bytes).decode('utf-8')
        
        return JSONResponse({
            "status": "success",
            "image": f"data:image/png;base64,{img_b64}",
            "params": {
                "whiteness": whiteness,
                "alignment": alignment,
                "timesteps": timesteps,
            }
        })

    except TimeoutError:
        raise HTTPException(status_code=504, detail=f"Hết thời gian xử lý ({PIPELINE_TIMEOUT}s)")
    except InferenceServerException as e:
        print(f"Interactive Triton Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)[:500]}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"System Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
