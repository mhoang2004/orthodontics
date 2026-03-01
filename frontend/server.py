from flask import Flask, send_from_directory, abort
import os

app = Flask(__name__)
# Đặt thư mục làm việc hiện tại là thư mục chứa server.py
# để Flask có thể tìm thấy các file static
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # Trả về các file như CSS, JS
    return send_from_directory(FRONTEND_DIR, filename)

if __name__ == '__main__':
    # Chạy Flask app trên cổng 5000
    app.run(host='0.0.0.0', port=5000, debug=True)