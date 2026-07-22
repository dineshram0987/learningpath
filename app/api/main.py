"""
app/api/main.py
----------------
FastAPI service for chart-type classification.

Endpoints:
  GET  /health                — liveness check
  POST /predict-chart-type    — upload an image, get chart class + confidence scores

Start:
    uvicorn app.api.main:app --reload --port 8000

Test via Swagger UI:
    http://localhost:8000/docs
"""

import io
import os
import sys
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "chart_classifier.pt")
IMG_SIZE     = 128
CHART_CLASSES = ["bar", "box", "histogram", "line", "pie", "scatter"]  # sorted (ImageFolder order)
DEVICE       = torch.device("cpu")          # API runs on CPU always
MAX_FILE_MB  = 10

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ChartPrediction(BaseModel):
    predicted_class: str
    confidence: float
    all_scores: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    classes: List[str]
    model_loaded: bool


# ── Model Definition (must match train_transfer.py) ───────────────────────────

def build_resnet18(num_classes: int) -> nn.Module:
    """Recreate the same ResNet18 head architecture used during training."""
    net = models.resnet18(weights=None)        # weights loaded from .pt file
    net.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(net.fc.in_features, num_classes),
    )
    return net


# ── Image Pre-processing ──────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes to a (1, 3, H, W) tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)    # add batch dim
    return tensor.to(DEVICE)


# ── App Lifecycle ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chart-Type Classifier API",
    description=(
        "Personalized Learning Platform — Day 3 FastAPI\n\n"
        "Classifies learner-submitted chart images into one of 6 categories: "
        "bar, box, histogram, line, pie, scatter."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state (loaded once at startup)
_state: Dict = {"model": None, "loaded": False}


@app.on_event("startup")
async def load_model():
    """Load the production model once at API startup."""
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python src/train_transfer.py` to generate it. "
            "API will start but /predict-chart-type will return 503."
        )
        _state["loaded"] = False
        return

    try:
        model = build_resnet18(num_classes=len(CHART_CLASSES))
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        model.eval()
        _state["model"] = model
        _state["loaded"] = True
        logger.info(f"[API] Loaded chart classifier from {MODEL_PATH}")
    except Exception as exc:
        logger.error(f"[API] Failed to load model: {exc}")
        _state["loaded"] = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Liveness check — returns model status and available classes."""
    return HealthResponse(
        status="ok",
        model="ResNet18 fine-tuned (ImageNet → Charts)",
        device=str(DEVICE),
        classes=CHART_CLASSES,
        model_loaded=_state["loaded"],
    )


@app.post("/predict-chart-type", response_model=ChartPrediction, tags=["Prediction"])
async def predict_chart_type(file: UploadFile = File(...)):
    """
    Upload a PNG/JPG chart image to classify its type.

    Returns:
      - **predicted_class**: Top predicted class (bar, line, scatter, pie, histogram, box)
      - **confidence**: Probability of the top class
      - **all_scores**: Softmax probabilities for all 6 classes
    """
    if not _state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train_transfer.py` first."
        )

    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail=f"Expected an image file, got {file.content_type}")

    # Read & validate size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File too large. Max {MAX_FILE_MB}MB.")

    try:
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422,
                            detail=f"Could not decode image: {exc}")

    with torch.no_grad():
        logits = _state["model"](tensor)          # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]  # (num_classes,)

    top_idx = int(probs.argmax().item())
    all_scores = {cls: round(float(probs[i].item()), 6)
                  for i, cls in enumerate(CHART_CLASSES)}

    logger.info(f"[API] Predicted: {CHART_CLASSES[top_idx]} "
                f"({probs[top_idx]:.3f}) for file={file.filename}")

    return ChartPrediction(
        predicted_class=CHART_CLASSES[top_idx],
        confidence=round(float(probs[top_idx].item()), 6),
        all_scores=all_scores,
    )
