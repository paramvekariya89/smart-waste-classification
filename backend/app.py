"""
Smart Waste Classification - Web Backend API
Subject: IPA (Image Processing and Applications)
Model: YOLOv11n exported to ONNX
Architecture: Hosted on Render with full CORS support for Vercel Frontend
"""

import os
import time
import base64
import random
import logging
from typing import Dict, Any, List

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS


# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("SmartWasteBackend")


# ------------------------------------------------------------------------------
# Flask & CORS Initialization
# ------------------------------------------------------------------------------
app = Flask(__name__)

# Allow Vercel frontend, localhost, and other clients
CORS(app, resources={r"/*": {"origins": "*"}})

# 16 MB maximum upload size
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ------------------------------------------------------------------------------
# Model Paths & Configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")

SAMPLES_DIR = os.path.join(
    BASE_DIR,
    "sample_images"
)

CONFIDENCE_THRESHOLD = 0.25

IOU_THRESHOLD = 0.45

IMAGE_SIZE = 320

MAX_DETECTIONS = 50


# ------------------------------------------------------------------------------
# Class-specific colors
# ------------------------------------------------------------------------------
CLASS_COLORS = {

    "E_waste": {
        "hex": "#0288D1",
        "bgr": (209, 136, 2)
    },

    "Medical_waste": {
        "hex": "#E53935",
        "bgr": (53, 57, 229)
    },

    "Hazardous_waste": {
        "hex": "#FB8C00",
        "bgr": (0, 140, 251)
    },

    "Chemical_waste": {
        "hex": "#8E24AA",
        "bgr": (170, 36, 142)
    },

    "Plastic_waste": {
        "hex": "#43A047",
        "bgr": (71, 160, 67)
    },

    "Paper_waste": {
        "hex": "#D97706",
        "bgr": (6, 119, 217)
    },

    "Unknown": {
        "hex": "#757575",
        "bgr": (117, 117, 117)
    }
}


# ------------------------------------------------------------------------------
# Model Classes
# ------------------------------------------------------------------------------
model_classes = {

    0: "E_waste",
    1: "Medical_waste",
    2: "Hazardous_waste",
    3: "Chemical_waste",
    4: "Plastic_waste",
    5: "Paper_waste"

}


# ------------------------------------------------------------------------------
# Domain-specific Eco-Disposal & Recycling Guidelines
# ------------------------------------------------------------------------------
DISPOSAL_GUIDELINES = {

    "E_waste": {
        "title": "Electronic Waste (E-Waste)",
        "action": (
            "Send to an authorized e-waste collection center "
            "or certified electronic recyclers."
        ),
        "caution": (
            "Do NOT burn, dismantle carelessly, "
            "or dispose with ordinary household trash."
        ),
        "benefit": (
            "Facilitates recovery of precious metals "
            "(gold, copper) while preventing lead and cadmium contamination."
        )
    },

    "Medical_waste": {
        "title": "Bio-Medical / Clinical Waste",
        "action": (
            "Segregate immediately into puncture-resistant, "
            "certified Red or Yellow biohazard waste containers."
        ),
        "caution": (
            "Never mix with regular municipal garbage. "
            "Extreme biological and infectious contamination risk."
        ),
        "benefit": (
            "Subjected to specialized clinical incineration "
            "or high-pressure autoclaving for sterilization."
        )
    },

    "Hazardous_waste": {
        "title": "Hazardous Domestic / Industrial Waste",
        "action": (
            "Store securely in sealed, leak-proof containers "
            "and take to official hazardous collection facilities."
        ),
        "caution": (
            "Keep away from heat, flame, and moisture. "
            "Do NOT dump into sewer systems or landfills."
        ),
        "benefit": (
            "Neutralizes heavy metals (lead-acid batteries, mercury vapor lamps) "
            "from leaching into groundwater."
        )
    },

    "Chemical_waste": {
        "title": "Chemical Waste & Industrial Reagents",
        "action": (
            "Neutralize using specialized chemical spill kits "
            "and absorbent mats; wear protective PPE gear."
        ),
        "caution": (
            "Never pour toxic chemicals, solvents, "
            "or strong acids down domestic drains or soil."
        ),
        "benefit": (
            "Ensures regulated chemical neutralization "
            "and high-temperature thermal treatment."
        )
    },

    "Plastic_waste": {
        "title": "Plastic Waste (Polymers)",
        "action": (
            "Rinse clean of food and chemical residues, "
            "flatten containers, and place in recyclable plastics bin."
        ),
        "caution": (
            "Never burn plastics, as open combustion "
            "generates toxic dioxins and furans."
        ),
        "benefit": (
            "Enables sorting (PET, HDPE, PP) and mechanical pelletizing "
            "into high-grade recycled plastic pellets."
        )
    },

    "Paper_waste": {
        "title": "Paper & Cardboard Waste",
        "action": (
            "Flatten corrugated cardboard boxes and place dry sheets "
            "in the designated clean paper recycling bin."
        ),
        "caution": (
            "Keep strictly dry. Heavily soiled, oil-stained, "
            "or wax-coated paper cannot be pulped."
        ),
        "benefit": (
            "Saves timber resources and reduces water/energy consumption "
            "in commercial papermaking."
        )
    },

    "None": {
        "title": "No Waste Detected",
        "action": (
            "No recognizable waste object was detected."
        ),
        "caution": (
            "Ensure proper lighting, clear focus, "
            "and place the waste item centrally within the camera frame."
        ),
        "benefit": (
            "Try capturing another angle or selecting a clearer image."
        )
    }
}


# ------------------------------------------------------------------------------
# Disclaimer
# ------------------------------------------------------------------------------
DISCLAIMER = (
    "AI predictions are intended for project demonstration "
    "and should not replace official waste-handling procedures."
)


# ------------------------------------------------------------------------------
# ONNX Runtime Model
# ------------------------------------------------------------------------------
onnx_session = None


def load_onnx_model():
    """
    Loads the exported YOLOv11n ONNX model using ONNX Runtime.
    """

    global onnx_session

    if not os.path.exists(MODEL_PATH):

        logger.error(
            "ONNX model '%s' not found.",
            MODEL_PATH
        )

        return False

    try:

        logger.info(
            "Loading ONNX model: %s",
            MODEL_PATH
        )

        onnx_session = ort.InferenceSession(
            MODEL_PATH,
            providers=[
                "CPUExecutionProvider"
            ]
        )

        logger.info(
            "ONNX model loaded successfully."
        )

        logger.info(
            "ONNX providers: %s",
            onnx_session.get_providers()
        )

        # Log input information
        for inp in onnx_session.get_inputs():

            logger.info(
                "ONNX input: name=%s shape=%s type=%s",
                inp.name,
                inp.shape,
                inp.type
            )

        # Log output information
        for output in onnx_session.get_outputs():

            logger.info(
                "ONNX output: name=%s shape=%s type=%s",
                output.name,
                output.shape,
                output.type
            )

        return True

    except Exception as e:

        logger.exception(
            "Failed to load ONNX model: %s",
            e
        )

        onnx_session = None

        return False


# ------------------------------------------------------------------------------
# Initialize model at startup
# ------------------------------------------------------------------------------
load_onnx_model()


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:

    _, ext = os.path.splitext(
        filename.lower()
    )

    return ext in ALLOWED_EXTENSIONS


def decode_image_from_bytes(
    image_bytes: bytes
) -> np.ndarray:

    """
    Safely decodes image bytes into a BGR OpenCV numpy array.
    """

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise ValueError(
            "Decoded image is empty or format is unsupported."
        )

    return img


def format_class_name(
    raw_name: str
) -> str:

    """
    Formats 'Plastic_waste' into 'Plastic Waste'.
    """

    return raw_name.replace(
        "_",
        " "
    ).title()


# ------------------------------------------------------------------------------
# YOLO Letterbox Preprocessing
# ------------------------------------------------------------------------------
def letterbox_image(
    image: np.ndarray,
    new_shape=(320, 320),
    color=(114, 114, 114)
):
    """
    Resizes image using YOLO-style letterboxing.

    Returns:
        resized_image
        scale_ratio
        padding (dw, dh)
    """

    original_shape = image.shape[:2]

    original_height = original_shape[0]
    original_width = original_shape[1]

    if isinstance(new_shape, int):

        new_shape = (
            new_shape,
            new_shape
        )

    target_height = new_shape[0]
    target_width = new_shape[1]

    ratio = min(
        target_height / original_height,
        target_width / original_width
    )

    new_unpad = (
        int(round(original_width * ratio)),
        int(round(original_height * ratio))
    )

    dw = target_width - new_unpad[0]
    dh = target_height - new_unpad[1]

    dw /= 2
    dh /= 2

    if (
        original_width,
        original_height
    ) != new_unpad:

        image = cv2.resize(
            image,
            new_unpad,
            interpolation=cv2.INTER_LINEAR
        )

    top = int(
        round(dh - 0.1)
    )

    bottom = int(
        round(dh + 0.1)
    )

    left = int(
        round(dw - 0.1)
    )

    right = int(
        round(dw + 0.1)
    )

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return image, ratio, (dw, dh)


# ------------------------------------------------------------------------------
# Non-Maximum Suppression
# ------------------------------------------------------------------------------
def nms_boxes(
    boxes,
    scores,
    iou_threshold=0.45
):

    """
    NumPy implementation of Non-Maximum Suppression.
    """

    if len(boxes) == 0:

        return []

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    widths = np.maximum(
        0,
        x2 - x1
    )

    heights = np.maximum(
        0,
        y2 - y1
    )

    areas = widths * heights

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:

        current = order[0]

        keep.append(
            int(current)
        )

        if order.size == 1:

            break

        xx1 = np.maximum(
            x1[current],
            x1[order[1:]]
        )

        yy1 = np.maximum(
            y1[current],
            y1[order[1:]]
        )

        xx2 = np.minimum(
            x2[current],
            x2[order[1:]]
        )

        yy2 = np.minimum(
            y2[current],
            y2[order[1:]]
        )

        width = np.maximum(
            0,
            xx2 - xx1
        )

        height = np.maximum(
            0,
            yy2 - yy1
        )

        intersection = width * height

        union = (
            areas[current]
            + areas[order[1:]]
            - intersection
            + 1e-7
        )

        iou = intersection / union

        remaining = np.where(
            iou <= iou_threshold
        )[0]

        order = order[
            remaining + 1
        ]

    return keep


# ------------------------------------------------------------------------------
# Direct ONNX Runtime Inference
# ------------------------------------------------------------------------------
def run_onnx_inference(
    image_bgr: np.ndarray
):

    """
    Runs YOLOv11n inference directly through ONNX Runtime.

    The exported model has the expected output format:

        (1, 10, 2100)

    where:

        4 values = bounding box
        6 values = class scores
    """

    if onnx_session is None:

        raise RuntimeError(
            "ONNX model is not loaded."
        )

    original_height, original_width = (
        image_bgr.shape[:2]
    )

    # --------------------------------------------------------------------------
    # 1. YOLO-style letterbox preprocessing
    # --------------------------------------------------------------------------
    image, ratio, (dw, dh) = letterbox_image(
        image_bgr,
        new_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    # --------------------------------------------------------------------------
    # 2. BGR -> RGB
    # --------------------------------------------------------------------------
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------------------------
    # 3. HWC -> CHW
    # --------------------------------------------------------------------------
    image = image.transpose(
        2,
        0,
        1
    )

    # --------------------------------------------------------------------------
    # 4. Convert to float32 and normalize
    # --------------------------------------------------------------------------
    image = np.ascontiguousarray(
        image,
        dtype=np.float32
    )

    image /= 255.0

    # --------------------------------------------------------------------------
    # 5. Add batch dimension
    # --------------------------------------------------------------------------
    input_tensor = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------------------------------------------------
    # 6. Get ONNX input name
    # --------------------------------------------------------------------------
    input_name = onnx_session.get_inputs()[0].name

    # --------------------------------------------------------------------------
    # 7. Direct ONNX Runtime inference
    # --------------------------------------------------------------------------
    outputs = onnx_session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    if not outputs:

        raise RuntimeError(
            "ONNX Runtime returned no output."
        )

    predictions = outputs[0]

    logger.info(
        "Raw ONNX output shape: %s",
        predictions.shape
    )

    # --------------------------------------------------------------------------
    # 8. Remove batch dimension
    # --------------------------------------------------------------------------
    if predictions.ndim == 3:

        predictions = predictions[0]

    # Expected:
    #
    # (10, 2100)
    #
    # Convert to:
    #
    # (2100, 10)
    #
    if predictions.ndim != 2:

        raise RuntimeError(
            f"Unexpected ONNX output dimensions: {predictions.shape}"
        )

    if predictions.shape[0] < predictions.shape[1]:

        predictions = predictions.transpose(
            1,
            0
        )

    # --------------------------------------------------------------------------
    # 9. Process predictions
    # --------------------------------------------------------------------------
    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:

        if len(prediction) < 5:

            continue

        # YOLO output:
        #
        # cx, cy, width, height, class_scores...
        #
        cx = float(prediction[0])
        cy = float(prediction[1])
        width = float(prediction[2])
        height = float(prediction[3])

        class_scores = prediction[4:]

        if len(class_scores) == 0:

            continue

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )

        # Confidence filtering
        if confidence < CONFIDENCE_THRESHOLD:

            continue

        # ----------------------------------------------------------------------
        # Convert xywh -> xyxy
        # ----------------------------------------------------------------------
        x1 = cx - (width / 2.0)
        y1 = cy - (height / 2.0)
        x2 = cx + (width / 2.0)
        y2 = cy + (height / 2.0)

        # ----------------------------------------------------------------------
        # Remove letterbox padding
        # ----------------------------------------------------------------------
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        # ----------------------------------------------------------------------
        # Clip boxes to original image
        # ----------------------------------------------------------------------
        x1 = max(
            0,
            min(
                original_width - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                original_height - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                original_width - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                original_height - 1,
                y2
            )
        )

        # Invalid box
        if x2 <= x1 or y2 <= y1:

            continue

        boxes.append(
            [
                x1,
                y1,
                x2,
                y2
            ]
        )

        scores.append(
            confidence
        )

        class_ids.append(
            class_id
        )

    # --------------------------------------------------------------------------
    # 10. Class-aware NMS
    # --------------------------------------------------------------------------
    final_indices = []

    if boxes:

        boxes_np = np.asarray(
            boxes,
            dtype=np.float32
        )

        scores_np = np.asarray(
            scores,
            dtype=np.float32
        )

        class_ids_np = np.asarray(
            class_ids,
            dtype=np.int32
        )

        unique_classes = np.unique(
            class_ids_np
        )

        for class_id in unique_classes:

            class_indices = np.where(
                class_ids_np == class_id
            )[0]

            class_boxes = boxes_np[
                class_indices
            ]

            class_scores = scores_np[
                class_indices
            ]

            keep_local = nms_boxes(
                class_boxes,
                class_scores,
                IOU_THRESHOLD
            )

            for local_index in keep_local:

                final_indices.append(
                    int(
                        class_indices[local_index]
                    )
                )

    # --------------------------------------------------------------------------
    # 11. Build frontend-compatible detections
    # --------------------------------------------------------------------------
    detections = []

    for index in final_indices:

        cls_id = int(
            class_ids[index]
        )

        confidence = (
            float(scores[index])
            * 100.0
        )

        bbox = [
            int(
                round(
                    boxes[index][0]
                )
            ),
            int(
                round(
                    boxes[index][1]
                )
            ),
            int(
                round(
                    boxes[index][2]
                )
            ),
            int(
                round(
                    boxes[index][3]
                )
            )
        ]

        cls_name = model_classes.get(
            cls_id,
            f"Class_{cls_id}"
        )

        detections.append({

            "class_id": cls_id,

            "class_name": cls_name,

            "class_display": format_class_name(
                cls_name
            ),

            "confidence": round(
                confidence,
                1
            ),

            "color_hex": CLASS_COLORS.get(
                cls_name,
                CLASS_COLORS["Unknown"]
            )["hex"],

            "bbox": bbox

        })

    # Highest confidence first
    detections.sort(
        key=lambda d: d["confidence"],
        reverse=True
    )

    # Limit maximum detections
    detections = detections[
        :MAX_DETECTIONS
    ]

    return detections


# ------------------------------------------------------------------------------
# Image Annotation
# ------------------------------------------------------------------------------
def annotate_image(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]]
) -> str:

    """
    Draws custom colored bounding boxes, labels, and confidence tags.
    Returns base64 JPEG.
    """

    annotated = image_bgr.copy()

    height, width = annotated.shape[:2]

    # Scaling factor for text and lines
    scale = max(
        0.45,
        min(width, height) / 900.0
    )

    thickness = max(
        2,
        int(scale * 2.5)
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        cls_name = det["class_name"]

        confidence = det["confidence"]

        color_info = CLASS_COLORS.get(
            cls_name,
            CLASS_COLORS["Unknown"]
        )

        color_bgr = color_info["bgr"]

        # ----------------------------------------------------------------------
        # Bounding box
        # ----------------------------------------------------------------------
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color_bgr,
            thickness
        )

        # ----------------------------------------------------------------------
        # Label
        # ----------------------------------------------------------------------
        label_text = (
            f"{cls_name} {confidence:.1f}%"
        )

        font_scale = max(
            0.4,
            scale * 0.8
        )

        (
            text_w,
            text_h
        ), baseline = cv2.getTextSize(
            label_text,
            font,
            font_scale,
            1
        )

        box_y1 = max(
            0,
            y1 - text_h - 10
        )

        box_y2 = y1

        box_x2 = min(
            width,
            x1 + text_w + 12
        )

        # If label would go outside image
        if y1 - text_h - 10 < 0:

            box_y1 = y1

            box_y2 = min(
                height,
                y1 + text_h + 10
            )

        # Label background
        cv2.rectangle(
            annotated,
            (x1, box_y1),
            (box_x2, box_y2),
            color_bgr,
            -1
        )

        if y1 - text_h - 10 < 0:

            text_y = box_y2 - 5

        else:

            text_y = y1 - 5

        # Label text
        cv2.putText(
            annotated,
            label_text,
            (x1 + 6, text_y),
            font,
            font_scale,
            (255, 255, 255),
            max(
                1,
                int(thickness / 2)
            ),
            cv2.LINE_AA
        )

    # --------------------------------------------------------------------------
    # Encode annotated image
    # --------------------------------------------------------------------------
    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        88
    ]

    success, buffer = cv2.imencode(
        ".jpg",
        annotated,
        encode_params
    )

    if not success:

        raise RuntimeError(
            "Failed to encode annotated image to JPEG buffer."
        )

    encoded_str = base64.b64encode(
        buffer
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded_str
    )


# ------------------------------------------------------------------------------
# API - Home
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():

    """
    Root information endpoint.
    """

    return jsonify({

        "service": "Smart Waste Classification API",

        "model": "YOLOv11n ONNX",

        "status": "online",

        "endpoints": {

            "health": "/health",

            "predict": "/predict (POST)",

            "sample": "/api/sample (GET)",

            "sample_by_class": "/api/sample/<category> (GET)"

        }

    })


# ------------------------------------------------------------------------------
# API - Health
# ------------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():

    """
    Returns backend and ONNX model health status.
    """

    return jsonify({

        "status": (
            "ready"
            if onnx_session is not None
            else "model_missing"
        ),

        "model": "YOLOv11n",

        "weights": "best.onnx",

        "confidence_threshold": CONFIDENCE_THRESHOLD,

        "classes_count": len(
            model_classes
        ),

        "classes": model_classes

    })


# ------------------------------------------------------------------------------
# API - Predict
# ------------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    """
    Receives image upload or base64 frame,
    executes direct ONNX Runtime inference,
    and returns detection results.
    """

    global onnx_session

    # --------------------------------------------------------------------------
    # Ensure model is loaded
    # --------------------------------------------------------------------------
    if onnx_session is None:

        if not load_onnx_model():

            return jsonify({

                "success": False,

                "error": (
                    "Trained YOLO model (best.onnx) "
                    "is not loaded or missing on server."
                )

            }), 503

    image_bytes = None

    # --------------------------------------------------------------------------
    # 1. Multipart/form-data upload
    # --------------------------------------------------------------------------
    if "image" in request.files:

        file = request.files["image"]

        if file.filename == "":

            return jsonify({

                "success": False,

                "error": (
                    "No file selected for analysis."
                )

            }), 400

        if not allowed_file(
            file.filename
        ):

            return jsonify({

                "success": False,

                "error": (
                    "Unsupported file format. "
                    "Please upload JPG, PNG, WEBP, "
                    "or BMP images."
                )

            }), 400

        image_bytes = file.read()

    # --------------------------------------------------------------------------
    # 2. JSON base64 image
    # --------------------------------------------------------------------------
    elif (
        request.is_json
        and "image_base64" in request.json
    ):

        raw_b64 = request.json[
            "image_base64"
        ]

        if "," in raw_b64:

            raw_b64 = raw_b64.split(
                ",",
                1
            )[1]

        try:

            image_bytes = base64.b64decode(
                raw_b64
            )

        except Exception:

            return jsonify({

                "success": False,

                "error": (
                    "Invalid base64 image data."
                )

            }), 400

    # --------------------------------------------------------------------------
    # No image
    # --------------------------------------------------------------------------
    if not image_bytes:

        return jsonify({

            "success": False,

            "error": (
                "No image data received "
                "in the request."
            )

        }), 400

    # --------------------------------------------------------------------------
    # Decode image
    # --------------------------------------------------------------------------
    try:

        bgr_image = decode_image_from_bytes(
            image_bytes
        )

    except Exception as e:

        logger.exception(
            "Image decoding failed: %s",
            e
        )

        return jsonify({

            "success": False,

            "error": (
                "Failed to decode image: "
                + str(e)
            )

        }), 400

    # --------------------------------------------------------------------------
    # Direct ONNX Runtime inference
    # --------------------------------------------------------------------------
    try:

        start_time = time.time()

        detections = run_onnx_inference(
            bgr_image
        )

        inference_time_ms = round(
            (
                time.time()
                - start_time
            ) * 1000.0,
            1
        )

        logger.info(
            "ONNX inference completed in %.1f ms with %d detection(s)",
            inference_time_ms,
            len(detections)
        )

    except Exception as e:

        logger.exception(
            "ONNX inference failed: %s",
            e
        )

        return jsonify({

            "success": False,

            "error": (
                "ONNX inference failed: "
                + str(e)
            )

        }), 500

    # --------------------------------------------------------------------------
    # Sort detections
    # --------------------------------------------------------------------------
    total_objects = len(
        detections
    )

    detections.sort(
        key=lambda d: d["confidence"],
        reverse=True
    )

    # --------------------------------------------------------------------------
    # Primary detection
    # --------------------------------------------------------------------------
    if total_objects > 0:

        primary_det = detections[0]

        primary_class = (
            primary_det["class_name"]
        )

        primary_class_display = (
            primary_det["class_display"]
        )

        primary_confidence = (
            primary_det["confidence"]
        )

        primary_color_hex = (
            primary_det["color_hex"]
        )

        guidelines = DISPOSAL_GUIDELINES.get(
            primary_class,
            DISPOSAL_GUIDELINES["None"]
        )

        status_message = (
            f"Successfully detected "
            f"{total_objects} object(s)."
        )

    else:

        primary_class = "None"

        primary_class_display = (
            "No Waste Detected"
        )

        primary_confidence = None

        primary_color_hex = (
            CLASS_COLORS["Unknown"]["hex"]
        )

        guidelines = (
            DISPOSAL_GUIDELINES["None"]
        )

        status_message = (
            "No waste object detected."
        )

    # --------------------------------------------------------------------------
    # Generate annotated image
    # --------------------------------------------------------------------------
    try:

        annotated_image_b64 = annotate_image(
            bgr_image,
            detections
        )

    except Exception as e:

        logger.exception(
            "Failed to render annotated image: %s",
            e
        )

        success, buffer = cv2.imencode(
            ".jpg",
            bgr_image
        )

        if success:

            annotated_image_b64 = (
                "data:image/jpeg;base64,"
                + base64.b64encode(
                    buffer
                ).decode("utf-8")
            )

        else:

            annotated_image_b64 = None

    # --------------------------------------------------------------------------
    # Final response
    # --------------------------------------------------------------------------
    return jsonify({

        "success": True,

        "message": status_message,

        "primary_class": primary_class,

        "primary_class_display": (
            primary_class_display
        ),

        "primary_confidence": (
            primary_confidence
        ),

        "primary_color_hex": (
            primary_color_hex
        ),

        "object_count": total_objects,

        "inference_time_ms": (
            inference_time_ms
        ),

        "detections": detections,

        "guidelines": guidelines,

        "disclaimer": DISCLAIMER,

        "annotated_image": (
            annotated_image_b64
        )

    })


# ------------------------------------------------------------------------------
# API - Sample Image
# ------------------------------------------------------------------------------
@app.route(
    "/api/sample",
    methods=["GET"]
)
@app.route(
    "/api/sample/<category>",
    methods=["GET"]
)
def get_sample(category=None):

    """
    Returns a curated sample image
    from sample_images/ directory.
    """

    if not os.path.exists(
        SAMPLES_DIR
    ):

        return jsonify({

            "success": False,

            "error": (
                "Sample images directory "
                "not found on server."
            )

        }), 404

    all_files = [

        f

        for f in os.listdir(
            SAMPLES_DIR
        )

        if f.lower().endswith(
            (
                ".jpg",
                ".png",
                ".jpeg"
            )
        )

    ]

    if not all_files:

        return jsonify({

            "success": False,

            "error": (
                "No sample images found."
            )

        }), 404

    # --------------------------------------------------------------------------
    # Select category-specific image
    # --------------------------------------------------------------------------
    if category:

        matched = [

            f

            for f in all_files

            if category.lower() in f.lower()

        ]

        if matched:

            selected = random.choice(
                matched
            )

        else:

            selected = random.choice(
                all_files
            )

    else:

        selected = random.choice(
            all_files
        )

    # --------------------------------------------------------------------------
    # Read image
    # --------------------------------------------------------------------------
    file_path = os.path.join(
        SAMPLES_DIR,
        selected
    )

    with open(
        file_path,
        "rb"
    ) as f:

        data = f.read()

    b64 = base64.b64encode(
        data
    ).decode("utf-8")

    # --------------------------------------------------------------------------
    # Return sample
    # --------------------------------------------------------------------------
    return jsonify({

        "success": True,

        "filename": selected,

        "image_base64": (
            "data:image/jpeg;base64,"
            + b64
        )

    })


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    logger.info(
        "Starting Smart Waste Classification "
        "Backend API on port %d...",
        port
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )