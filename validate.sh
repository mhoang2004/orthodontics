#!/bin/bash
set -e

echo "=== Server health ==="
curl -sf http://localhost:8000/v2/health/ready && echo "OK" || echo "FAIL"

echo "=== Model ready ==="
curl -sf http://localhost:8000/v2/models/dental_pipeline/ready && echo "OK" || echo "FAIL"

echo "=== Model config ==="
curl -s http://localhost:8000/v2/models/dental_pipeline | python3 -m json.tool

pip install tritonclient[http] -q

python3 - <<'EOF'
import tritonclient.http as httpclient
import numpy as np, cv2, json, time, sys

client = httpclient.InferenceServerClient("localhost:8000")

with open("/home/trislord/Dev/Python/orthodontics/Data/case1.jpg", "rb") as f:
    raw_bytes = f.read()
    img = np.array([raw_bytes], dtype=object)

inputs = [httpclient.InferInput("IMAGE", img.shape, "BYTES")]
inputs[0].set_data_from_numpy(img)
outputs = [httpclient.InferRequestedOutput("PRED_IMAGE", binary_data=True),
           httpclient.InferRequestedOutput("METADATA")]

t0 = time.time()
result = client.infer("dental_pipeline", inputs, outputs=outputs, timeout=120)
elapsed = time.time() - t0

pred = cv2.imdecode(
    np.frombuffer(result.as_numpy("PRED_IMAGE").flat[0], np.uint8),
    cv2.IMREAD_COLOR)
meta = json.loads(result.as_numpy("METADATA").flat[0])

assert pred is not None,       "FAIL: pred_image is None"
assert pred.ndim == 3,         f"FAIL: wrong shape {pred.shape}"
assert pred.dtype == np.uint8, f"FAIL: wrong dtype {pred.dtype}"

cv2.imwrite("/tmp/validation_result.png", pred)

print(f"PASS — shape:{meta['shape']}  time:{elapsed:.1f}s")
print(f"Saved: /tmp/validation_result.png")
EOF

python3 - <<'EOF'
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient("localhost:8000")
bad = np.array([b"this is not an image"], dtype=object)
inputs = [httpclient.InferInput("IMAGE", bad.shape, "BYTES")]
inputs[0].set_data_from_numpy(bad)
try:
    client.infer("dental_pipeline", inputs, timeout=30)
    print("NOTE: bad input did not raise on client side")
except Exception as e:
    print(f"PASS — bad input returned error (not server crash): {type(e).__name__}")

import urllib.request
urllib.request.urlopen("http://localhost:8000/v2/health/ready")
print("PASS — server still alive after bad input")
EOF

echo "=== Temp files after inference ==="
docker exec dental_triton ls /tmp/triton_io/ | wc -l

echo "=== GPU memory after inference ==="
docker exec dental_triton nvidia-smi --query-gpu=memory.used,memory.free \
  --format=csv,noheader
