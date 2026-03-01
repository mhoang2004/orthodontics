# Frontend - Ứng dụng dự đoán chỉnh nha

Giao diện web đơn giản cho người dùng upload ảnh và nhận kết quả dự đoán chỉnh nha.

## 📁 Cấu trúc

```
frontend/
├── index.html       # Giao diện chính
├── style.css        # Styling (trước/sau comparison)
├── script.js        # Logic client-side (upload, gọi API)
├── server.py        # Flask server để serve frontend
└── README.md        # File này
```

## 🚀 Chạy Frontend

### Cách 1: Dùng Flask server (Recommended)

```powershell
# 1. Cài Flask và dependencies
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D
pip install flask

# 2. Chạy Flask server
cd frontend
python server.py

# 3. Mở trình duyệt
# Truy cập: http://localhost:5000
```

### Cách 2: Dùng Python built-in HTTP server

```powershell
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D\frontend
python -m http.server 5000
# Mở: http://localhost:5000
```

### Cách 3: Mở trực tiếp (không cần server)

```powershell
# Mở file trong trình duyệt
# File: D:\nckh2025\3D-guided-Tooth-Alignment-2D\frontend\index.html
# Hoặc dùng: start index.html
```

## ⚙️ Chạy toàn bộ hệ thống (Backend + Frontend)

Bạn cần chạy **3 services** song song:

### Terminal 1: Chạy Backend API (FastAPI)

```powershell
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D

# Nếu chưa có virtual env
python -m venv .venv

# Activate
.venv\Scripts\Activate.ps1

# Install backend requirements
pip install -r backend/requirements.txt

# Chạy backend (port 8001)
cd backend
uvicorn app:app --host 0.0.0.0 --port 8001
```

**Output kỳ vọng:**
```
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### Terminal 2: Chạy Frontend Server

```powershell
cd D:\nckh2025\3D-guided-Tooth-Alignment-2D\frontend

# Nếu chưa cài Flask
pip install flask

# Chạy Flask (port 5000)
python server.py
```

**Output kỳ vọng:**
```
 * Running on http://0.0.0.0:5000
```

### Terminal 3: (Optional) Triton Server

Nếu bạn muốn chạy Triton thay vì Backend API trực tiếp:

```powershell
$ROOT="D:/nckh2025/3D-guided-Tooth-Alignment-2D"
docker run --rm -it `
  -p8000:8000 -p8001:8001 -p8002:8002 `
  -v "${ROOT}/triton_model_repository":/models `
  -v "${ROOT}/Code":/workspace/Code `
  -e PIPELINE_CODE_DIR=/workspace/Code `
  nvcr.io/nvidia/tritonserver:23.08-py3 `
  tritonserver --model-repository=/models
```

Sau đó cập nhật backend API URL từ 8001 → 8000 nếu cần.

## 🌐 Sử dụng

1. **Mở frontend**: http://localhost:5000
2. **Upload ảnh**: Kéo thả hoặc nhấp chọn ảnh JPG/PNG
3. **Chờ xử lý**: Tùy vào hiệu năng máy (1-5 phút nếu dùng CPU)
4. **Xem kết quả**: So sánh ảnh trước/sau
5. **Tải kết quả**: Nhấp nút "Tải kết quả"

## 🔧 Cấu hình

### Đổi Backend URL (nếu khác localhost:8001)

Mở console trình duyệt (F12 → Console) và chạy:

```javascript
localStorage.setItem('backendUrl', 'http://your-backend-ip:8001');
location.reload();
```

Ví dụ (nếu backend chạy trên máy khác):

```javascript
localStorage.setItem('backendUrl', 'http://192.168.1.100:8001');
```

## 📋 Tính năng

✅ Upload ảnh (drag & drop)
✅ Hiển thị trước/sau (so sánh side-by-side)
✅ Tải kết quả PNG
✅ Status API real-time
✅ Responsive design (mobile-friendly)
✅ Hướng dẫn trong ứng dụng

## ⚠️ Lỗi thường gặp

### "Không kết nối Backend"

```
❌ Không kết nối
```

**Giải pháp:**
- Kiểm tra backend chạy trên port 8001: `python -m uvicorn app:app --host 0.0.0.0 --port 8001`
- Nếu backend ở máy khác, set lại URL: `localStorage.setItem('backendUrl', 'http://ip:8001')`

### "Processing timeout (504)"

**Giải pháp:**
- Tăng timeout ở backend: `set PIPELINE_TIMEOUT=3600` (1 giờ)
- Hoặc tăng timeout browser (nếu dùng nginx reverse-proxy)

### CORS Error

**Giải pháp:**
- Dùng Flask server (xử lý tự động)
- Hoặc cài `python-multipart` nếu cần fix CORS backend

## 📞 Support

Nếu có lỗi:

1. Kiểm tra console browser (F12)
2. Kiểm tra logs backend (Terminal 1)
3. Kiểm tra logs frontend server (Terminal 2)

---

Logo: 🦷 Tooth Alignment
Đề tài: Dự đoán trước và sau khi chỉnh nha
Tác giả: Nghiên cứu khoa học sinh viên 2026
