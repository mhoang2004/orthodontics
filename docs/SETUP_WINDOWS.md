# 🚀 Hướng dẫn chạy toàn bộ hệ thống (Windows)

Hướng dẫn chi tiết cách cài đặt và chạy Backend API + Frontend trên Windows.

## 📋 Yêu cầu

- Windows 10/11
- Python 3.7+ (khuyên dùng 3.9 trở lên)
- Kết nối internet
- ~10GB RAM (tùy vào model size)

## 🔧 Cài đặt môi trường (lần đầu)

### Bước 1: Clone/Tải project

```powershell
# Nếu chưa có project, tải về
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D
```

### Bước 2: Tạo Python Virtual Environment

```powershell
# Mở PowerShell, điều hướng đến workspace
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D

# Tạo virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Nếu gặp lỗi "execution policy", chạy:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Output kỳ vọng:**
```
(.venv) PS D:\nckh2025\3D-guided-Tooth-Alignment-2D>
```

### Bước 3: Cài đặt toàn bộ dependencies

```powershell
# Đảm bảo ở trong virtual environment (.venv)
pip install --upgrade pip

# Cài dependencies của project Code/
pip install -r Code/requirements.txt

# Cài dependencies của Backend API
pip install -r backend/requirements.txt

# Cài Flask cho frontend server
pip install flask
```

**Nếu gặp lỗi cài PyTorch:**

Nếu máy có GPU NVIDIA:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Nếu chỉ CPU:
```powershell
pip install torch torchvision torchaudio
```

## ▶️ Chạy toàn bộ hệ thống

Bạn cần mở **2-3 PowerShell terminals** và chạy song song:

### Terminal 1: Chạy Backend API

```powershell
# PowerShell Terminal 1
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Chạy backend (port 8001)
cd backend
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

**Output kỳ vọng:**
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Application startup complete
```

**Kiểm tra backend**: Mở trình duyệt → http://localhost:8001/docs (Swagger UI)

### Terminal 2: Chạy Frontend Server

```powershell
# PowerShell Terminal 2
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D\frontend

# Chạy Flask server (port 5000)
python server.py
```

**Output kỳ vọng:**
```
 * Serving Flask app 'server'
 * Debug mode: on
 * Running on http://0.0.0.0:5000
```

**Mở frontend**: http://localhost:5000

### Terminal 3: (Optional) Triton Server

Nếu bạn muốn dùng Triton thay vì backend API trực tiếp:

```powershell
# PowerShell Terminal 3 (dùng Docker)
$ROOT="D:/nckh2025/3D-guided-Tooth-Alignment-2D"
docker run --rm -it `
  -p8000:8000 -p8001:8001 -p8002:8002 `
  -v "${ROOT}/triton_model_repository":/models `
  -v "${ROOT}/Code":/workspace/Code `
  -e PIPELINE_CODE_DIR=/workspace/Code `
  nvcr.io/nvidia/tritonserver:23.08-py3 `
  tritonserver --model-repository=/models
```

**Lưu ý** Triton cần Docker Desktop cài trên Windows.

## 🌐 Sử dụng ứng dụng

1. **Mở frontend**:
   ```
   http://localhost:5000
   ```

2. **Upload ảnh**: Kéo thả hoặc nhấp chọn ảnh JPG/PNG

3. **Chờ xử lý**: Ứng dụng sẽ gọi backend API để xử lý (1-10 phút tùy máy)

4. **Xem kết quả**: So sánh ảnh trước/sau

5. **Tải kết quả**: Nhấp "Tải kết quả"

## ⏱️ Tăng timeout (nếu xử lý lâu)

Nếu gặp "504 Gateway Timeout", tăng timeout:

```powershell
# Trước khi chạy Backend API, set env var:
$env:PIPELINE_TIMEOUT="3600"

# Hoặc dùng lệnh trực tiếp trong Terminal 1:
# Trước khi chạy uvicorn
```

## 🔍 Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'torch'"

```powershell
pip install torch torchvision torchaudio
```

### ❌ "ImportError: DLL load failed" (PyTorch)

- Cài Microsoft Visual C++ Redistributable: https://support.microsoft.com/en-us/help/2977003
- Hoặc cài PyTorch CPU version

### ❌ Backend "Could not locate Code directory"

```powershell
# Set environment variable
$env:PIPELINE_CODE_DIR = "D:\nckh2025\3D-guided-Tooth-Alignment-2D\Code"

# Rồi chạy uvicorn
uvicorn app:app --host 0.0.0.0 --port 8001
```

### ❌ Frontend không kết nối Backend

Mở console browser (F12) → Console → chạy:

```javascript
// Check backend URL
console.log('Backend URL:', localStorage.getItem('backendUrl') || 'http://localhost:8001');

// Nếu khác, set custom URL
localStorage.setItem('backendUrl', 'http://your-ip:8001');
location.reload();
```

### ❌ Port bị chiếm

Nếu port 5000 hoặc 8001 đang dùng:

```powershell
# Tìm process dùng port
netstat -ano | findstr :8001

# Kill process (PID là số ở cuối dòng)
taskkill /PID <PID> /F

# Hoặc chạy trên port khác
uvicorn app:app --host 0.0.0.0 --port 8002
```

## 📁 Cấu trúc thư mục

```
D:\nckh2025\3D-guided-Tooth-Alignment-2D\
├── Code/                          # Project code (Stage1, Stage2, Stage3, etc.)
├── backend/                       # Backend API (FastAPI)
│   ├── app.py
│   └── requirements.txt
├── frontend/                      # Frontend (Vue.js)
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── server.py                 # Flask server
│   └── README.md
├── triton_model_repository/       # Triton model repo (optional)
├── Output/                        # Output results (auto-created)
└── .venv/                        # Virtual environment
```

## 📝 Log & Debug

### Xem logs Backend

Logs sẽ hiện ở Terminal 1 (backend). Kiểm tra lỗi:

```
ERROR: Error uploading image
Traceback...
```

### Xem logs Frontend

Mở DevTools (F12) → Console → kiểm tra:

```
Frontend initialized. Backend URL: http://localhost:8001
```

## 🎯 Kiểm tra toàn bộ

```powershell
# 1. Check Python version
python --version

# 2. Check virtual env
.venv\Scripts\Activate.ps1
pip list | grep -E "fastapi|uvicorn|torch|flask"

# 3. Check port availability
netstat -ano | findstr :5000
netstat -ano | findstr :8001

# 4. Test backend
curl -X GET http://localhost:8001

# 5. Test frontend
Start-Process "http://localhost:5000"
```

## 🚀 Quick Start (lần sau)

```powershell
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D
.venv\Scripts\Activate.ps1

# Terminal 1
cd backend && uvicorn app:app --host 0.0.0.0 --port 8001

# Terminal 2
cd frontend && python server.py

# Mở browser
Start-Process "http://localhost:5000"
```

## 💡 Tips

- **Lần đầu chậm**: Model lần đầu sẽ load checkpoint từ disk, chậm ~1-2 phút
- **Lần sau nhanh**: Sau khi loaded, infer sẽ nhanh hơn
- **GPU vs CPU**: Nếu có GPU NVIDIA, đảm bảo PyTorch dùng CUDA để nhanh 5-10x
- **Memory**: Nếu RAM < 8GB, có thể chậm. Đóng các ứng dụng khác

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra logs ở cả 3 terminals
2. Kiểm tra DevTools (F12) ở frontend
3. Kiểm tra Port có bị chiếm không
4. Kiểm tra PIPELINE_TIMEOUT nếu timeout

---

**Happy Tooth Alignment! 🦷**
