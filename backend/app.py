from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess, uuid, os
import aiofiles
import io

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

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # 3. Xử lý tên file và định dạng
    uid = str(uuid.uuid4())
    
    # Lấy extension gốc của file khách gửi (.jpg, .png, .jpeg, ...)
    original_ext = os.path.splitext(file.filename)[1].lower()
    if original_ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ định dạng JPG hoặc PNG")

    # Tạo đường dẫn lưu file đầu vào tạm thời với đúng extension
    inp_name = f"input_{uid}"
    inp_path = os.path.abspath(os.path.join(ROOT, f"{inp_name}{original_ext}"))
    
    # Giả định: main.py luôn xuất ra file .png trong thư mục prediction
    expected_main_output_path = os.path.abspath(os.path.join(MAIN_OUTPUT_DIR, f"{inp_name}.png"))

    try:
        # 4. Lưu file upload bất đồng bộ
        async with aiofiles.open(inp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # 5. Gọi Pipeline xử lý (main.py)
        print(f"--- Đang xử lý ảnh: {inp_path} ---")
        # Chạy subprocess tại thư mục Code để main.py tìm được các module liên quan
        cmd = ["python", "main.py", "-i", inp_path]
        
        proc = subprocess.run(
            cmd, 
            cwd=CODE_DIR, 
            capture_output=True, 
            text=True, 
            timeout=PIPELINE_TIMEOUT
        )

        # Kiểm tra lỗi khi chạy script
        if proc.returncode != 0:
            error_msg = proc.stderr if proc.stderr else "Lỗi không xác định trong main.py"
            print(f"Pipeline Error: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Pipeline error: {error_msg[:500]}")

        # 6. Kiểm tra kết quả đầu ra
        if not os.path.exists(expected_main_output_path):
            stdout_tail = proc.stdout[-500:] if proc.stdout else "No stdout"
            print(f"Không tìm thấy kết quả tại: {expected_main_output_path}")
            raise HTTPException(
                status_code=500, 
                detail=f"Kết quả không tồn tại. Kiểm tra log main.py: {stdout_tail}"
            )

        # 7. Trả về kết quả cho Client
        # Đọc file kết quả và stream về dưới dạng ảnh PNG
        async with aiofiles.open(expected_main_output_path, "rb") as f:
            file_bytes = await f.read()
        
        return StreamingResponse(
            io.BytesIO(file_bytes), 
            media_type="image/png", 
            headers={"Content-Disposition": f"attachment; filename=result_{uid}.png"}
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail=f"Hết thời gian xử lý ({PIPELINE_TIMEOUT}s)")
    except Exception as e:
        print(f"System Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
    
    finally:
        # 8. Dọn dẹp: Xóa file input tạm sau khi xong (giữ lại output để debug nếu cần)
        if os.path.exists(inp_path):
            try:
                os.remove(inp_path)
            except Exception as e:
                print(f"Cảnh báo: Không thể xóa file tạm {inp_path}: {e}")
