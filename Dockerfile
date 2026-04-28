# Stage 1: Build dlib wheel
FROM nvcr.io/nvidia/tritonserver:22.08-py3 AS dlib-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /wheels \
    && pip install --no-cache-dir "numpy<2.0.0" \
    && export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    && pip wheel --no-cache-dir --wheel-dir /wheels --no-build-isolation dlib==19.24.2

# Stage 2: Final Image
FROM nvcr.io/nvidia/tritonserver:22.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install torch with extra index url based on requirements.txt
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dlib from built wheel
COPY --from=dlib-builder /wheels /wheels
RUN pip install /wheels/dlib-*.whl && rm -rf /wheels

# Install other dependencies
RUN pip install pyyaml opencv-python-headless diffusers transformers accelerate "numpy<2.0.0" scikit-image natsort matplotlib Pillow tqdm

COPY Code/ /app/src/

ENV DENTAL_CONFIG_PATH=/app/src/Config.yaml
ENV TRITON_TMP_DIR=/tmp/triton_io
ENV PYTHONPATH=/app:/app/src

RUN mkdir -p /tmp/triton_io

CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1", "--log-info=true", "--log-warning=true", "--log-error=true"]
