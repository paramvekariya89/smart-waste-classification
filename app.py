"""
Smart Waste Classification - Web Application (Flask Backend)
Subject: IPA (Image Processing and Applications)
Model: YOLOv11n (Ultralytics)
"""

import os
import io
import time
import base64
import gc
import logging
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmartWasteApp")

# ------------------------------------------------------------------------------
# Flask Initialization
# ------------------------------------------------------------------------------
app = Flask(__name__)
# 16 MB max upload size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# ------------------------------------------------------------------------------
# Model Paths & Configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
CONFIDENCE_THRESHOLD = 0.25

# Class-specific colors (Hex and BGR for OpenCV rendering)
CLASS_COLORS = {
    "E_waste": {"hex": "#0288D1", "bgr": (209, 136, 2)},
    "Medical_waste": {"hex": "#E53935", "bgr": (53, 57, 229)},
    "Hazardous_waste": {"hex": "#FB8C00", "bgr": (0, 140, 251)},
    "Chemical_waste": {"hex": "#8E24AA", "bgr": (170, 36, 142)},
    "Plastic_waste": {"hex": "#43A047", "bgr": (71, 160, 67)},
    "Paper_waste": {"hex": "#D97706", "bgr": (6, 119, 217)},
    "Unknown": {"hex": "#757575", "bgr": (117, 117, 117)}
}

# Domain-specific Eco-Disposal & Recycling Guidelines
DISPOSAL_GUIDELINES = {
    "E_waste": {
        "title": "Electronic Waste (E-Waste)",
        "action": "Send to an authorized e-waste collection center or certified electronic recyclers.",
        "caution": "Do NOT burn, dismantle carelessly, or dispose with ordinary household trash.",
        "benefit": "Facilitates recovery of precious metals (gold, copper) while preventing lead and cadmium contamination."
    },
    "Medical_waste": {
        "title": "Bio-Medical / Clinical Waste",
        "action": "Segregate immediately into puncture-resistant, certified Red or Yellow biohazard waste containers.",
        "caution": "Never mix with regular municipal garbage. Extreme biological and infectious contamination risk.",
        "benefit": "Subjected to specialized clinical incineration or high-pressure autoclaving for sterilization."
    },
    "Hazardous_waste": {
        "title": "Hazardous Domestic / Industrial Waste",
        "action": "Store securely in sealed, leak-proof containers and take to official hazardous collection facilities.",
        "caution": "Keep away from heat, flame, and moisture. Do NOT dump into sewer systems or landfills.",
        "benefit": "Neutralizes heavy metals (lead-acid batteries, mercury vapor lamps) from leaching into groundwater."
    },
    "Chemical_waste": {
        "title": "Chemical Waste & Industrial Reagents",
        "action": "Neutralize using specialized chemical spill kits and absorbent mats; wear protective PPE gear.",
        "caution": "Never pour toxic chemicals, solvents, or strong acids down domestic drains or soil.",
        "benefit": "Ensures regulated chemical neutralization and high-temperature thermal treatment."
    },
    "Plastic_waste": {
        "title": "Plastic Waste (Polymers)",
        "action": "Rinse clean of food and chemical residues, flatten containers, and place in recyclable plastics bin.",
        "caution": "Never burn plastics, as open combustion generates toxic dioxins and furans.",
        "benefit": "Enables sorting (PET, HDPE, PP) and mechanical pelletizing into high-grade recycled plastic pellets."
    },
    "Paper_waste": {
        "title": "Paper & Cardboard Waste",
        "action": "Flatten corrugated cardboard boxes and place dry sheets in the designated clean paper recycling bin.",
        "caution": "Keep strictly dry. Heavily soiled, oil-stained, or wax-coated paper cannot be pulped.",
        "benefit": "Saves timber resources and reduces water/energy consumption in commercial papermaking."
    },
    "None": {
        "title": "No Waste Detected",
        "action": "No recognizable waste object was detected.",
        "caution": "Ensure proper lighting, clear focus, and place the waste item centrally within the camera frame.",
        "benefit": "Try capturing another angle or selecting a clearer image."
    }
}

DISCLAIMER = "AI predictions are intended for project demonstration and should not replace official waste-handling procedures."

# ------------------------------------------------------------------------------
# Model Singleton Loader
# ------------------------------------------------------------------------------
yolo_model = None
model_classes = {}

def load_yolo_model():
    """Loads YOLOv11n model once globally and validates classes."""
    global yolo_model, model_classes

    if not os.path.exists(MODEL_PATH):
        logger.error("Trained model weights '%s' not found.", MODEL_PATH)
        return False

    try:
        yolo_model = YOLO(MODEL_PATH)
        # Render deployment: explicitly keep YOLO on CPU to avoid GPU/CUDA initialization.
        yolo_model.to("cpu")
        if hasattr(yolo_model, "names") and isinstance(yolo_model.names, dict):
            model_classes = yolo_model.names
        else:
            model_classes = {i: name for i, name in enumerate(yolo_model.names)}
        logger.info("Model loaded successfully")
        logger.info("Model classes: %s", model_classes)
        logger.info("Inference device: CPU | imgsz: 320 | max_det: 10")
        return True
    except Exception as e:
        logger.exception("Failed to load YOLO model: %s", e)
        return False

# Load on module load
load_yolo_model()

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

def decode_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Safely decodes image bytes into a BGR OpenCV numpy array."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Decoded image is empty or format is unsupported.")
    return img

def format_class_name(raw_name: str) -> str:
    """Formats 'Plastic_waste' into 'Plastic Waste'."""
    return raw_name.replace("_", " ").title()

def annotate_image(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    """Draws custom colored bounding boxes, labels, and confidence tags, returning base64 JPEG."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    # Scaling factor for text and lines based on image resolution
    scale = max(0.45, min(w, h) / 900.0)
    thickness = max(2, int(scale * 2.5))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class_name"]
        conf = det["confidence"]

        color_info = CLASS_COLORS.get(cls_name, CLASS_COLORS["Unknown"])
        color_bgr = color_info["bgr"]

        # 1. Main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)

        # 2. Label text badge
        label_text = f"{cls_name} {conf:.1f}%"
        font_scale = max(0.4, scale * 0.8)
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

        # Top label background rectangle
        box_y1 = max(0, y1 - text_h - 10)
        box_y2 = y1
        box_x2 = min(w, x1 + text_w + 12)

        # If box is too close to top edge, place inside the box
        if y1 - text_h - 10 < 0:
            box_y1 = y1
            box_y2 = y1 + text_h + 10

        cv2.rectangle(annotated, (x1, box_y1), (box_x2, box_y2), color_bgr, -1)
        text_y = box_y2 - 5 if y1 - text_h - 10 < 0 else y1 - 5

        # White contrasting text
        cv2.putText(
            annotated,
            label_text,
            (x1 + 6, text_y),
            font,
            font_scale,
            (255, 255, 255),
            max(1, int(thickness / 2)),
            cv2.LINE_AA
        )

    # Encode to JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 88]
    success, buffer = cv2.imencode('.jpg', annotated, encode_params)
    if not success:
        raise RuntimeError("Failed to encode annotated image to JPEG buffer.")

    encoded_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_str}"

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serves the main web dashboard."""
    return render_template(
        "index.html",
        model_ready=(yolo_model is not None),
        classes=list(model_classes.values()) if model_classes else []
    )

@app.route("/health", methods=["GET"])
def health():
    """Returns server and model health status."""
    return jsonify({
        "status": "ready" if yolo_model is not None else "model_missing",
        "model": "YOLOv11n",
        "weights": "best.pt",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "classes_count": len(model_classes),
        "classes": model_classes
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Receives image upload, executes YOLOv11n inference, and returns detection results."""
    global yolo_model
    if yolo_model is None:
        # Attempt to reload if it wasn't loaded initially
        if not load_yolo_model():
            return jsonify({
                "success": False,
                "error": "Trained YOLO model (best.pt) is not loaded or missing on the server."
            }), 503

    # Check incoming file
    image_bytes = None

    # Handle standard multipart/form-data file upload
    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected for analysis."}), 400
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported file format. Please upload JPG, PNG, WEBP, or BMP images."
            }), 400
        image_bytes = file.read()

    # Handle JSON base64 payload (e.g. from camera capture)
    elif request.is_json and "image_base64" in request.json:
        raw_b64 = request.json["image_base64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw_b64)
        except Exception:
            return jsonify({"success": False, "error": "Invalid base64 image data."}), 400

    if not image_bytes:
        return jsonify({"success": False, "error": "No image data received in the request."}), 400

    # Decode image using OpenCV
    try:
        bgr_image = decode_image_from_bytes(image_bytes)
    except Exception as e:
        logger.warning("Corrupted image received: %s", e)
        return jsonify({"success": False, "error": "Failed to decode image. File may be corrupted or unreadable."}), 400

    # Execute Ultralytics inference
    # Render deployment: use CPU + smaller inference size to reduce RAM usage.
    try:
        # Release temporary Python/OpenCV memory before inference.
        gc.collect()

        # Limit very large uploaded images before YOLO processing.
        # This reduces memory usage while preserving the image aspect ratio.
        max_dimension = 1024
        image_h, image_w = bgr_image.shape[:2]
        largest_dimension = max(image_h, image_w)

        if largest_dimension > max_dimension:
            resize_scale = max_dimension / float(largest_dimension)
            new_width = max(1, int(image_w * resize_scale))
            new_height = max(1, int(image_h * resize_scale))
            bgr_image = cv2.resize(
                bgr_image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            logger.info(
                "Resized input image from %sx%s to %sx%s for lower-memory inference.",
                image_w,
                image_h,
                new_width,
                new_height
            )

        start_time = time.time()

        results = yolo_model.predict(
            bgr_image,
            conf=CONFIDENCE_THRESHOLD,
            device="cpu",
            imgsz=320,
            max_det=10,
            verbose=False
        )

        inference_time_ms = round((time.time() - start_time) * 1000.0, 1)
        logger.info("YOLO inference completed in %.1f ms", inference_time_ms)

        # Give Python/OpenCV a chance to release temporary inference memory.
        gc.collect()

    except Exception as e:
        logger.exception("Inference execution failed: %s", e)
        return jsonify({
            "success": False,
            "error": "An error occurred during YOLO model inference."
        }), 500

    detections = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item()) * 100.0
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

            # Resolve class name from model.names
            if cls_id in model_classes:
                cls_name = model_classes[cls_id]
            else:
                cls_name = f"Class_{cls_id}"

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "class_display": format_class_name(cls_name),
                "confidence": round(conf_val, 1),
                "color_hex": CLASS_COLORS.get(cls_name, CLASS_COLORS["Unknown"])["hex"],
                "bbox": xyxy
            })

    total_objects = len(detections)

    # Sort detections descending by confidence
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    if total_objects > 0:
        primary_det = detections[0]
        primary_class = primary_det["class_name"]
        primary_class_display = primary_det["class_display"]
        primary_confidence = primary_det["confidence"]
        primary_color_hex = primary_det["color_hex"]
        guidelines = DISPOSAL_GUIDELINES.get(primary_class, DISPOSAL_GUIDELINES["None"])
        status_message = f"Successfully detected {total_objects} object(s)."
    else:
        primary_class = "None"
        primary_class_display = "No Waste Detected"
        primary_confidence = None
        primary_color_hex = CLASS_COLORS["Unknown"]["hex"]
        guidelines = DISPOSAL_GUIDELINES["None"]
        status_message = "No waste object detected."

    # Generate annotated image with bounding boxes
    try:
        annotated_image_b64 = annotate_image(bgr_image, detections)
    except Exception as e:
        logger.exception("Failed to render annotated image: %s", e)
        # Fallback: re-encode original image without boxes
        _, buf = cv2.imencode('.jpg', bgr_image)
        annotated_image_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    return jsonify({
        "success": True,
        "message": status_message,
        "primary_class": primary_class,
        "primary_class_display": primary_class_display,
        "primary_confidence": primary_confidence,
        "primary_color_hex": primary_color_hex,
        "object_count": total_objects,
        "inference_time_ms": inference_time_ms,
        "detections": detections,
        "guidelines": guidelines,
        "disclaimer": DISCLAIMER,
        "annotated_image": annotated_image_b64
    })


@app.route("/api/sample", methods=["GET"])
def get_sample():
    """Picks a random test image from the test dataset (mirroring Tkinter 'Test Sample' feature)."""
    import random
    test_dirs = [
        os.path.join(BASE_DIR, "Dataset", "Final_merged_dataset", "merged", "test", "images"),
        os.path.join(BASE_DIR, "Dataset", "Final_merged_dataset", "test", "images"),
    ]
    
    found_dir = None
    for d in test_dirs:
        if os.path.exists(d):
            found_dir = d
            break
            
    if not found_dir and os.path.exists(os.path.join(BASE_DIR, "Dataset")):
        # Check subdirectories
        for sub in os.listdir(os.path.join(BASE_DIR, "Dataset")):
            candidate = os.path.join(BASE_DIR, "Dataset", sub, "test", "images")
            if os.path.exists(candidate):
                found_dir = candidate
                break

    if not found_dir:
        return jsonify({"success": False, "error": "No test dataset folder found on server."}), 404

    image_files = [f for f in os.listdir(found_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        return jsonify({"success": False, "error": "No images found in test folder."}), 404

    selected = random.choice(image_files)
    file_path = os.path.join(found_dir, selected)
    
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode('utf-8')

    return jsonify({
        "success": True,
        "filename": selected,
        "image_base64": f"data:image/jpeg;base64,{b64}"
    })


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Smart Waste Classification Web Application on port %d...", port)
    app.run(host="0.0.0.0", port=port, debug=False)
