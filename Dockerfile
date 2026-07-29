# ─────────────────────────────────────────────────────────────────────────────
# CUDA 12.3 + cuDNN 9 — matches onnxruntime-gpu 1.23.x
# No TensorFlow, no DeepFace, no retina-face
# ─────────────────────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
# build-essential  → compiles insightface Cython extension
# libgl1           → OpenCV imshow / imwrite
# libglib2.0-0     → OpenCV threading internals
# libgomp1         → ONNX Runtime parallel ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3

# ── Python deps ───────────────────────────────────────────────────────────────
# requirements.txt is copied alone so this heavy layer is cached
# independently from source code changes.
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt \
 && python -c "import numpy; print('NumPy version inside docker build:', numpy.__version__)" \
 && pip freeze

# ── Pre-download InsightFace buffalo_sc model weights ─────────────────────────
# buffalo_sc = SCRFD-500M detector + 5-point landmark head (pure ONNX, no TF).
# Downloading at build time prevents a cold-start network hit on first request.
# ctx_id=-1 forces CPU so no GPU is needed during the build stage.
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1, det_size=(640, 640)); \
print('InsightFace buffalo_sc weights ready')" \
|| echo "InsightFace pre-download failed — will retry at runtime"

# ── Copy AdaFace ONNX model ───────────────────────────────────────────────────
# Copied before full source so this layer is cached separately.
# Only re-runs if adaface_ir101.onnx itself changes.
COPY app/models/adaface_ir101.onnx app/models/adaface_ir101.onnx

# ── Application source ────────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]