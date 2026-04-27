# 1. Tạo và kích hoạt môi trường conda mới
conda create -n ortho_py39 python=3.9 -y
conda activate ortho_py39

# 2. Cài đặt dlib
conda install -c conda-forge dlib -y

# 3. Cài Pytorch với CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 4. Cài các package còn lại
pip install -r requirements.txt

## chạy backend
uvicorn backend.app:app --host 0.0.0.0 --port 8001

## chạy frontend
python server.py