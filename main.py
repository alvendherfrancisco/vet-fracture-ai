import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision
import warnings
import base64
import logging
import threading

from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Absolute path to the repo root — keeps FileResponse reliable regardless of cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 4
    IMAGE_SIZE = 1024

    CONF_THRESH = {
        1: 0.55,  # 32-A Simple
        2: 0.45,  # 32-B Wedge
        3: 0.55,  # 32-C Complex
    }
    AO_LABELS = {
        1: "32-A (Simple)",
        2: "32-B (Wedge)",
        3: "32-C (Complex)",
    }


HF_REPO_ID = "alvendherfrancisco/VetFractureAI-Model"
HF_FILENAME = "best_model.pth"


# =============================================================================
# BACKGROUND MODEL LOADING
# FastAPI starts immediately; the heavy download+load runs in a daemon thread.
# =============================================================================
_model = None
_model_loading = True
_model_error: str | None = None


def _init_model():
    global _model, _model_loading, _model_error
    try:
        if not os.path.exists(Config.MODEL_PATH):
            logger.info("Downloading model from Hugging Face …")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir=BASE_DIR,
                token=os.environ.get("HF_TOKEN"),
            )
            logger.info("Download complete.")

        logger.info("Loading model weights …")
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, Config.NUM_CLASSES)

        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        model.to(Config.DEVICE)
        model.eval()

        _model = model
        logger.info("Model ready on %s.", Config.DEVICE)
    except Exception as exc:
        _model_error = str(exc)
        logger.error("Model loading failed: %s", exc)
    finally:
        _model_loading = False


threading.Thread(target=_init_model, daemon=True).start()


def _check_model():
    """Returns a JSONResponse if the model isn't ready, else None."""
    if _model_loading:
        return JSONResponse(
            {"error": "loading", "message": "Model is still loading — please wait and try again."},
            status_code=503,
        )
    if _model_error:
        return JSONResponse(
            {"error": "failed", "message": f"Model failed to load: {_model_error}"},
            status_code=500,
        )
    return None


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=Config.IMAGE_SIZE,
            min_width=Config.IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


# =============================================================================
# CLAHE PREVIEW
# =============================================================================
def apply_clahe_preview(image_array):
    if image_array.ndim == 3:
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_gray)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)


# =============================================================================
# INFERENCE
# =============================================================================
def run_inference(image_array, conf_32a=0.55, conf_32b=0.45, conf_32c=0.55):
    Config.CONF_THRESH[1] = conf_32a
    Config.CONF_THRESH[2] = conf_32b
    Config.CONF_THRESH[3] = conf_32c

    orig_h, orig_w = image_array.shape[:2]

    # Convert to grayscale and back to RGB — matches the training pipeline on HF
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    image_array_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    augmented = get_transform()(image=image_array_gray)
    tensor_img = augmented["image"].unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        output = _model(tensor_img)[0]

    keep = torchvision.ops.nms(output["boxes"], output["scores"], iou_threshold=0.3)
    boxes  = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()

    scale = Config.IMAGE_SIZE / max(orig_h, orig_w)
    pad_y = (Config.IMAGE_SIZE - int(orig_h * scale)) // 2
    pad_x = (Config.IMAGE_SIZE - int(orig_w * scale)) // 2

    result_img = image_array.copy()
    detections = []

    for box, label, score in zip(boxes, labels, scores):
        label = int(label)
        if label == 0 or score < Config.CONF_THRESH.get(label, 0.5):
            continue

        x1 = max(0, int((box[0] - pad_x) / scale))
        y1 = max(0, int((box[1] - pad_y) / scale))
        x2 = min(orig_w, int((box[2] - pad_x) / scale))
        y2 = min(orig_h, int((box[3] - pad_y) / scale))

        class_name = Config.AO_LABELS[label]
        detections.append({"class": class_name, "confidence": round(float(score), 4), "box": [x1, y1, x2, y2]})

        color = (255, 193, 7) if label == 1 else (255, 159, 67) if label == 2 else (220, 53, 69)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 4)
        text = f"{class_name} {score:.1%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result_img, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
        cv2.putText(result_img, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result_img, detections


def numpy_to_base64(img_array):
    buf = BytesIO()
    Image.fromarray(img_array.astype(np.uint8)).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="VetFractureAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # HuggingFace Docker Space — same domain, wildcard is fine
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static frontend ──────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(BASE_DIR, "style.css"), media_type="text/css")

@app.get("/script.js")
async def serve_js():
    return FileResponse(os.path.join(BASE_DIR, "script.js"), media_type="application/javascript")


# ── Health / model status ────────────────────────────────────────────────────
@app.get("/health")
def health():
    if _model_loading:
        return JSONResponse({"status": "loading", "message": "Model is loading…"}, status_code=503)
    if _model_error:
        return JSONResponse({"status": "error",   "message": _model_error},          status_code=500)
    return {"status": "ok", "device": str(Config.DEVICE)}


# ── CLAHE ────────────────────────────────────────────────────────────────────
@app.post("/clahe")
async def clahe_endpoint(file: UploadFile = File(...)):
    if (err := _check_model()) is not None:
        return err
    try:
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Could not decode image."}, status_code=400)
        enhanced = apply_clahe_preview(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return JSONResponse({"image": numpy_to_base64(enhanced), "message": "CLAHE applied successfully."})
    except Exception as exc:
        logger.exception("CLAHE error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── Predict ──────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    conf_32a: float = Form(0.55),
    conf_32b: float = Form(0.45),
    conf_32c: float = Form(0.55),
):
    if (err := _check_model()) is not None:
        return err
    try:
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Could not decode image."}, status_code=400)
        result_img, detections = run_inference(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf_32a, conf_32b, conf_32c)
        summary = (
            f"Detected {len(detections)} fracture(s): "
            + ", ".join(f"{d['class']} ({d['confidence']:.0%})" for d in detections)
        ) if detections else "No fractures detected above confidence threshold."
        return JSONResponse({"image": numpy_to_base64(result_img), "detections": detections, "summary": summary, "total": len(detections)})
    except Exception as exc:
        logger.exception("Predict error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)