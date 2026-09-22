# ♻️ Smart Waste Classification System

### AI-Powered Waste Detection, Classification & Eco-Disposal Guidance using YOLOv11n

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Smart Waste Classification System** is an AI-powered computer vision project developed for **IPA (Image Processing and Applications)**. It uses a custom-trained **YOLOv11n** object detection model to identify different types of waste from images or camera input and provides classification results together with disposal and recycling guidance.

---

## 📌 Project Overview

Improper waste segregation and disposal can create environmental, health, and safety problems. The goal of this project is to demonstrate how **Artificial Intelligence and Computer Vision** can assist with automated waste identification.

The system uses a custom-trained **YOLOv11n (Nano)** model to detect and classify waste objects in an image.

The application provides:

* 🔍 Automatic waste detection
* 🎯 Bounding-box localization
* 📊 Confidence score for each detection
* ♻️ Waste classification
* 📷 Image upload
* 📹 Browser camera capture
* 🧠 YOLOv11n AI inference
* 📋 Object count and detection details
* 🌱 Disposal and recycling guidance
* 🌐 Web-based user interface
* 🖥️ Original Tkinter desktop application

---

## 🎯 Objectives

The main objectives of this project are:

1. To develop an AI-based waste detection system.
2. To train a YOLOv11n model for waste object detection.
3. To combine multiple waste datasets into a unified dataset.
4. To classify waste into predefined categories.
5. To display detected objects using bounding boxes.
6. To provide confidence scores for predictions.
7. To provide useful disposal and recycling recommendations.
8. To convert the original desktop application into a web-based application.
9. To create a system that can be accessed through a browser.

---

## 🏷️ Waste Categories

The trained model supports **six waste categories**:

| Class ID | Waste Category    | Description                          |
| -------: | ----------------- | ------------------------------------ |
|        0 | `E_waste`         | Electronic and electrical waste      |
|        1 | `Medical_waste`   | Biomedical and medical-related waste |
|        2 | `Hazardous_waste` | Waste requiring controlled handling  |
|        3 | `Chemical_waste`  | Chemical and spill-related waste     |
|        4 | `Plastic_waste`   | Plastic-based waste materials        |
|        5 | `Paper_waste`     | Paper and paper-based waste          |

> **Note:** The application uses the class names stored in the trained YOLO model as the primary source of truth.

---

## 🧠 Machine Learning Model

This project uses:

**YOLOv11n (YOLOv11 Nano)**

YOLO is a real-time object detection architecture capable of locating and classifying objects within images.

### Model

```text
Base Model:
YOLOv11n

Training Approach:
Transfer Learning

Model File:
best.pt

Task:
Object Detection

Input:
Image / Camera Frame

Output:
Bounding Boxes + Class + Confidence
```

The trained model is stored as:

```text
best.pt
```

The web application loads this model and performs inference whenever an image is submitted.

---

## 🔄 System Workflow

```text
                  ┌─────────────────────┐
                  │     User Input      │
                  │                     │
                  │  Image / Camera     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Flask Backend    │
                  │                     │
                  │ Image Processing    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    YOLOv11n Model   │
                  │                     │
                  │     best.pt         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     Detection       │
                  │                     │
                  │ Class + Confidence  │
                  │ Bounding Boxes      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Web Dashboard     │
                  │                     │
                  │ Results + Guidance │
                  └─────────────────────┘
```

---

## 🛠️ Technology Stack

### Machine Learning

* Python
* Ultralytics YOLOv11n
* PyTorch
* Transfer Learning

### Computer Vision

* OpenCV
* Pillow
* NumPy

### Backend

* Flask
* Werkzeug
* Python REST API

### Frontend

* HTML5
* CSS3
* JavaScript
* Browser MediaDevices API

### Deployment

* GitHub
* Vercel configuration
* Python WSGI entry point

---

## 📂 Project Structure

```text
smart-waste-classification/
│
├── api/
│   └── index.py
│       # Vercel/serverless WSGI entry point
│
├── static/
│   ├── style.css
│   │   # Web application styling
│   │
│   └── script.js
│       # Frontend logic, camera and API communication
│
├── templates/
│   └── index.html
│       # Main web dashboard
│
├── app.py
│   # Flask backend and YOLO inference API
│
├── app_gui.py
│   # Original Tkinter desktop application
│
├── best.pt
│   # Trained YOLOv11n model weights
│
├── train.py
│   # YOLO model training and validation script
│
├── merge_datasets.py
│   # Dataset merging and class mapping script
│
├── requirements.txt
│   # Python dependencies
│
├── vercel.json
│   # Deployment configuration
│
├── .gitignore
│   # Git ignored files and directories
│
└── README.md
    # Project documentation
```

---

# 🚀 Running the Web Application Locally

## 1. Clone the Repository

```bash
git clone https://github.com/paramvekariya89/smart-waste-classification.git
```

Move into the project directory:

```bash
cd smart-waste-classification
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start the Flask Application

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000
```

Open the address in your browser.

---

# 📷 Using the Web Application

## Method 1 — Upload an Image

1. Open the web application.
2. Select **Upload Image**.
3. Choose an image containing waste.
4. Click **Analyze Image**.
5. The YOLOv11n model processes the image.
6. Detection results are displayed on the dashboard.

The result includes:

* Detected class
* Confidence percentage
* Bounding box
* Number of detected objects
* Inference time
* Disposal guidance
* Annotated image

---

## Method 2 — Camera Detection

The web version uses the browser's camera interface.

1. Click **Use Camera**.
2. Allow camera access when requested.
3. Position the waste object in front of the camera.
4. Click **Capture Frame**.
5. The captured frame is sent to the backend.
6. YOLOv11n performs detection.
7. Results are displayed on the dashboard.

The browser camera approach allows the web application to use the user's camera without requiring the Python application to directly access the computer's webcam.

---

## Method 3 — Test Sample

When running the complete project locally with the test dataset available, the test functionality can load a sample image and perform inference.

For a deployed web application, the dataset itself is not required for normal user image prediction.

---

# 🔍 Detection Output

For every detected object, the application can provide information such as:

```text
Class:
Plastic Waste

Confidence:
94.7%

Object Count:
2

Inference Time:
32.4 ms
```

The image is also annotated with bounding boxes around detected objects.

Example response structure:

```json
{
    "success": true,
    "primary_class": "Plastic_waste",
    "primary_confidence": 94.7,
    "object_count": 2,
    "inference_time_ms": 32.4,
    "detections": [
        {
            "class_id": 4,
            "class_name": "Plastic_waste",
            "confidence": 94.7,
            "bbox": [120, 85, 450, 510]
        }
    ]
}
```

---

# 📡 API

The web application provides a prediction endpoint:

```text
POST /predict
```

## Request

The API accepts an image through multipart form data:

```text
image=<image file>
```

It can also process a Base64 image payload when supported by the frontend.

---

## Response

The API returns information including:

```text
success
message
primary_class
primary_confidence
object_cou_
```
