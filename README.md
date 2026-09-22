# ♻ SMART WASTE CLASSIFICATION SYSTEM

**AI-Powered Waste Detection & Eco-Disposal Guidance using YOLOv11n**  
*Subject: IPA (Image Processing and Applications)*

---

## 📌 Project Overview & Objective

The **Smart Waste Classification System** is an engineering AI application that performs real-time visual waste detection, localization, and classification into six primary categories using a custom-trained **YOLOv11n (Nano)** deep learning model. 

In addition to detecting objects and rendering bounding boxes with class confidence percentages, the system provides domain-specific **Eco-Disposal and Recycling Guidelines** to prevent improper handling of hazardous, chemical, bio-medical, and electronic waste.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, Flask, Werkzeug
- **Machine Learning**: Ultralytics YOLOv11n (`best.pt`), PyTorch
- **Computer Vision & Image Processing**: OpenCV (OpenCV Headless), Pillow, NumPy
- **Frontend**: HTML5, CSS3 (Modern AI Dashboard layout), Vanilla JavaScript (Webcam API via `navigator.mediaDevices.getUserMedia`)
- **Deployment**: Local WSGI, Vercel Serverless configuration (`vercel.json`, `api/index.py`)

---

## 🏷️ Supported Waste Categories (6 Classes)

The trained model (`best.pt`) maps to the following six verified classes:

| Class ID | Class Name | Category Color | Recommended Handling |
| :---: | :--- | :---: | :--- |
| `0` | **E_waste** | `#0288D1` (Blue) | Authorized E-Waste collection & metal recovery centers |
| `1` | **Medical_waste** | `#E53935` (Red) | Certified biohazard bins; clinical incineration |
| `2` | **Hazardous_waste** | `#FB8C00` (Orange) | Sealed containment; hazardous chemical depots |
| `3` | **Chemical_waste** | `#8E24AA` (Purple) | Spill neutralizers, PPE, regulated treatment |
| `4` | **Plastic_waste** | `#43A047` (Green) | Cleaned, sorted by polymer (PET/HDPE), pelletized |
| `5` | **Paper_waste** | `#D97706` (Amber) | Keep dry, flatten corrugated boxes, paper pulping |

---

## 📂 Project Structure

```text
Smart Waste Classification Web Version/
│
├── app.py                     # Flask backend API & YOLO inference engine
├── best.pt                    # Trained 6-class YOLOv11n weights (5.4 MB)
├── requirements.txt           # Minimal web inference dependencies
├── vercel.json                # Vercel serverless deployment config
├── .gitignore                 # Excludes datasets/runs/cache, keeps best.pt
├── README.md                  # Comprehensive documentation & viva guide
│
├── api/
│   └── index.py               # Vercel WSGI entry point
│
├── templates/
│   └── index.html             # Responsive modern AI dashboard
│
├── static/
│   ├── style.css              # Custom styling, dark navy header, status cards
│   └── script.js              # Camera stream, drag-drop, API communication
│
├── app_gui.py                 # Preserved original Tkinter desktop GUI
├── train.py                   # Preserved training script
├── merge_datasets.py          # Preserved dataset merger script
├── Dataset/                   # Dataset repository (excluded from git/deployment)
└── runs/                      # Training outputs & logs (excluded from git/deployment)
```

---

## 🚀 How to Install & Run Locally (Windows)

### 1. Open Terminal & Navigate to Project
```powershell
cd "e:\STUDY\NIRMA EI\SEM-5\IPA\Smart Waste Classification Web Version"
```

### 2. (Optional) Create and Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Flask Web Application
```powershell
python app.py
```

### 5. Access the Web Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 How to Test the Web Application

1. **Upload an Image**:
   - Click **Upload Image** or drag and drop any `.jpg`, `.png`, `.webp`, or `.bmp` file into the display area.
   - Click **Analyze Image** to run YOLOv11n inference.
2. **Camera Capture**:
   - Click **Use Camera** to activate your webcam.
   - Click **Capture Frame** to capture a snapshot and immediately detect waste.
3. **Quick Test Sample**:
   - Click **Test Sample** to load a random image directly from the test dataset and classify it.
4. **Clear Dashboard**:
   - Click **Clear** to reset the display screen, result badges, progress bar, and guidelines.

---

## 📡 Prediction API Specification

### `POST /predict`
Submits an image for YOLOv11n object detection and classification.

#### Request Formats:
- **Multipart Form Data**:
  - Key: `image` (File upload)
- **JSON Payload** (Base64 data URL):
  ```json
  {
    "image_base64": "data:image/jpeg;base64,..."
  }
  ```

#### Response Example:
```json
{
  "success": true,
  "message": "Successfully detected 2 object(s).",
  "primary_class": "Plastic_waste",
  "primary_class_display": "Plastic Waste",
  "primary_confidence": 94.7,
  "primary_color_hex": "#43A047",
  "object_count": 2,
  "inference_time_ms": 32.4,
  "detections": [
    {
      "class_id": 4,
      "class_name": "Plastic_waste",
      "class_display": "Plastic Waste",
      "confidence": 94.7,
      "color_hex": "#43A047",
      "bbox": [120, 85, 450, 510]
    }
  ],
  "guidelines": {
    "title": "Plastic Waste (Polymers)",
    "action": "Rinse clean of food and chemical residues...",
    "caution": "Never burn plastics, as open combustion generates toxic dioxins...",
    "benefit": "Enables sorting (PET, HDPE, PP) and mechanical pelletizing..."
  },
  "disclaimer": "AI predictions are intended for project demonstration...",
  "annotated_image": "data:image/jpeg;base64,..."
}
```

---

## ☁️ Vercel Deployment Guide & Architecture Notes

### Vercel Deployment Steps:
1. Ensure the repository is pushed to GitHub (with `Dataset/` and `runs/` excluded via `.gitignore`).
2. Log into [Vercel](https://vercel.com) and click **"Add New Project"**.
3. Import your GitHub repository.
4. Vercel automatically detects `vercel.json` and configures the Python serverless runtime.
5. Click **Deploy**.

### ⚠️ Important Serverless / Vercel Limitations:
- **Model Size vs. Library Size**: While `best.pt` is only **5.4 MB**, the required deep learning dependencies (`torch` + `ultralytics`) have an uncompressed size exceeding **500 MB**. Vercel's standard Serverless Function bundle size limit is **250 MB**.
- **Recommended Free Production Alternatives**:
  If Vercel build fails due to the 250 MB bundle limit, the standard recommended platforms for full PyTorch/Ultralytics hosting are:
  - **Hugging Face Spaces (Free / Gradio or Docker Flask)**: Specifically designed for ML models with PyTorch.
  - **Render.com / Railway (Docker Web Service)**: Supports persistent containers without serverless bundle restrictions.
  - **Google Cloud Run**: Serverless container execution with support for custom Docker images up to several gigabytes.

---

## 🎓 IPA Viva / Project Demonstration Workflow

During your viva/demonstration, explain the core dataflow:
```text
IMAGE INPUT (Upload / Webcam / Test Sample)
                   ↓
PRE-PROCESSING (In-Memory OpenCV BGR Decode)
                   ↓
YOLOv11n INFERENCE (best.pt, Conf Threshold = 0.25)
                   ↓
BOUNDING BOX & CONFIDENCE COMPUTATION
                   ↓
CLASS-SPECIFIC COLOR ANNOTATION
                   ↓
ECO-DISPOSAL & RECYCLING REASONING
```
