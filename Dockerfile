# ==========================================================
# STAGE 1: BUILDER (Biên dịch Dlib và các wheel C++ nặng)
# ==========================================================
ARG DLIB_WHEELS_IMAGE=dlib-builder

FROM python:3.10-slim AS dlib-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Cài đặt trình biên dịch hệ thống để build dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Biên dịch dlib thành định dạng Wheel trực tiếp ra thư mục /wheels
# Sử dụng Multi-threading (nproc) để build cực nhanh < 1 phút
RUN mkdir /wheels \
    && pip install --no-cache-dir numpy \
    && export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    && pip wheel --no-cache-dir --wheel-dir /wheels --no-build-isolation dlib==19.24.2

# Cho phép tái sử dụng wheel dlib từ image đã build sẵn.
# Mặc định dùng stage nội bộ "dlib-builder" nếu không truyền DLIB_WHEELS_IMAGE.
FROM ${DLIB_WHEELS_IMAGE} AS dlib-cache
# ==========================================================
# STAGE 2: TRITON RUNTIME BASE (AI Backend)
# ==========================================================
FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS triton-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Cài đặt các gói phần mềm bắt buộc cho trình biên dịch C++ (Cần thiết cho Dlib, OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Lấy wheel build sẵn từ stage nội bộ hoặc image cache bên ngoài
COPY --from=dlib-cache /wheels /wheels

# Copy requirements từ Folder Code/
COPY Code/requirements.txt /tmp/code-requirements.txt

# Cài PyTorch CUDA cho Python backend (model.py import torch tại thời điểm load model).
# Giữ lọc torch trong requirements để tránh xung đột phiên bản.
RUN grep -viE '^(torch|torchvision|torchaudio|dlib)\b' /tmp/code-requirements.txt > /tmp/code-requirements-notorch.txt \
    && pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
       torch==2.2.2 \
    && pip install --no-cache-dir -r /tmp/code-requirements-notorch.txt \
    && pip install --no-cache-dir --no-index --find-links=/wheels dlib \
    && rm -rf /wheels /tmp/code-requirements.txt /tmp/code-requirements-notorch.txt

# Tạo một thư mục chứa mã nguồn cho Python backend truy xuất tới
WORKDIR /app/Code
RUN mkdir -p /app/Code

# Khai báo các port của Triton
EXPOSE 8000 8001 8002

# Khởi chạy Triton Server chỉ định Model Repository mà ta sẽ mount vào trong Bước 3
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1", "--backend-config=python,shm-default-byte-size=67108864"]

# ==========================================================
# STAGE 3: DEV IMAGE (Có copy code để chạy không cần volume)
# ==========================================================
FROM triton-runtime AS triton-dev

# Dùng khi muốn chạy image độc lập không mount ./Code từ host.
COPY Code /app/Code
