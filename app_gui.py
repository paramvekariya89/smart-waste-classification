"""
Smart Waste Classification - Tkinter Desktop Application
Subject: IPA (Image Processing and Applications)
Target Model: YOLOv11n (Ultralytics)

Features:
- Large visual display screen on the left (Webcam stream / Uploaded Image / Inference results)
- 4 Primary Action Buttons on the right:
  1. Camera: Toggle live webcam feed on/off
  2. Capture: Freeze current camera frame, run YOLOv11n detection & display class
  3. Upload: Pick any image from disk and classify waste
  4. Test: Automatically load and test a random sample image from the test set
- Comprehensive Results Card:
  - Detected waste class badge
  - Confidence percentage
  - Object count & individual detection list
  - Waste disposal & recycling recommendations
"""

import os
import random
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

WINDOW_TITLE = "Smart Waste Classification System - YOLOv11n"
WINDOW_WIDTH = 1150
WINDOW_HEIGHT = 720

# Model weights path - FINAL TRAINED MODEL
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CUSTOM_WEIGHTS = os.path.join(
    PROJECT_DIR,
    "best.pt"
)

FALLBACK_WEIGHTS = "yolo11n.pt"

# Target classes
CLASS_NAMES = [
    "E_waste",
    "Medical_waste",
    "Hazardous_waste",
    "Chemical_waste",
    "Plastic_waste"
]

# Color palette for each waste class (Hex and BGR for OpenCV)
CLASS_COLORS = {
    "E_waste": {"hex": "#0288D1", "bgr": (209, 136, 2)},          # Blue
    "Medical_waste": {"hex": "#E53935", "bgr": (53, 57, 229)},     # Red
    "Hazardous_waste": {"hex": "#FB8C00", "bgr": (0, 140, 251)},   # Orange
    "Chemical_waste": {"hex": "#8E24AA", "bgr": (170, 36, 142)},   # Purple
    "Plastic_waste": {"hex": "#43A047", "bgr": (71, 160, 67)},     # Green
    "Unknown": {"hex": "#757575", "bgr": (117, 117, 117)}          # Gray
}

# Disposal & handling guidance for each waste category (Domain application for IPA project)
DISPOSAL_GUIDELINES = {
    "E_waste": (
        "Electronic Waste: Contains recyclable circuitry and hazardous heavy metals.\n"
        "• Action: Send to an authorized E-waste collection/recycling facility.\n"
        "• Caution: Do NOT burn or dispose in regular household waste."
    ),
    "Medical_waste": (
        "Biohazard / Medical Waste: High risk of infectious contamination.\n"
        "• Action: Dispose in certified Red/Yellow biohazard medical containers.\n"
        "• Caution: Requires high-temperature autoclaving or clinical incineration."
    ),
    "Hazardous_waste": (
        "Hazardous Waste (Batteries / Mercury lamps / Toxic elements).\n"
        "• Action: Store in leak-proof containers and take to hazardous disposal centers.\n"
        "• Caution: Highly toxic to groundwater and soil if dumped in landfills."
    ),
    "Chemical_waste": (
        "Chemical Waste / Spill: Corrosive, reactive, or flammable substances.\n"
        "• Action: Absorb using chemical spill neutralizers; use protective gear.\n"
        "• Caution: Never pour chemicals down domestic drainage systems."
    ),
    "Plastic_waste": (
        "Plastic Waste: Recyclable polymers (PET, HDPE, LDPE, PP).\n"
        "• Action: Clean out food/organic residue and place in Blue/Green recycling bin.\n"
        "• Benefit: Can be reprocessed into recycled plastic granules and fibers."
    ),
    "None": "No waste object detected in the image. Please try another angle or object."
}


# ==============================================================================
# MAIN APPLICATION CLASS
# ==============================================================================

class SmartWasteApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(980, 640)
        self.root.configure(bg="#F4F6F9")

        # Camera state
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None       # Stores current raw BGR image
        self.last_annotated_frame = None # Stores frame with bounding boxes drawn

        # Load YOLO Model
        self.load_model()

        # Build GUI Layout
        self.create_layout()

        # Handle window close cleanly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --------------------------------------------------------------------------
    # 1. Model Loading
    # --------------------------------------------------------------------------
    def load_model(self):
        """Loads trained custom model if available, otherwise falls back to yolo11n.pt."""
        if os.path.exists(CUSTOM_WEIGHTS):
            self.model_path = CUSTOM_WEIGHTS
            self.model_status_text = "Model: Trained YOLOv11n (best.pt)"
            print(f"[✓] Loading custom trained model: {CUSTOM_WEIGHTS}")
        elif os.path.exists(os.path.join("runs", "detect", "smart_waste_yolo11n", "weights", "best.pt")):
            self.model_path = os.path.join("runs", "detect", "smart_waste_yolo11n", "weights", "best.pt")
            self.model_status_text = "Model: Trained YOLOv11n (runs/detect/.../best.pt)"
            print(f"[✓] Loading model from runs: {self.model_path}")
        else:
            self.model_path = FALLBACK_WEIGHTS
            self.model_status_text = "Model: YOLOv11n Base (Untrained - Run train.py first)"
            print(f"[!] Custom weights '{CUSTOM_WEIGHTS}' not found. Loading base '{FALLBACK_WEIGHTS}'.")

        try:
            self.model = YOLO(self.model_path)
            print("[✓] Model initialized successfully.")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load YOLO model:\n{str(e)}")
            self.model = None

    # --------------------------------------------------------------------------
    # 2. GUI Layout
    # --------------------------------------------------------------------------
    def create_layout(self):
        """Sets up the left video/image display screen and right button/results panel."""
        # Top Header Bar
        header_frame = tk.Frame(self.root, bg="#1E293B", height=50)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="♻ SMART WASTE CLASSIFICATION SYSTEM (IPA - YOLOv11n)",
            font=("Segoe UI", 14, "bold"),
            bg="#1E293B",
            fg="#FFFFFF"
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=10)

        self.status_bar_lbl = tk.Label(
            header_frame,
            text=self.model_status_text,
            font=("Segoe UI", 10),
            bg="#1E293B",
            fg="#38BDF8"
        )
        self.status_bar_lbl.pack(side=tk.RIGHT, padx=20, pady=10)

        # Main Container (2 Columns: Left Screen, Right Controls)
        main_frame = tk.Frame(self.root, bg="#F4F6F9")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)

        # ----------------------------------------------------------------------
        # LEFT COLUMN: Visual Display Screen
        # ----------------------------------------------------------------------
        left_frame = tk.LabelFrame(
            main_frame,
            text=" Display Screen ",
            font=("Segoe UI", 11, "bold"),
            bg="#FFFFFF",
            fg="#334155",
            bd=2,
            relief=tk.GROOVE
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Canvas / Screen Label for rendering frames
        self.screen_lbl = tk.Label(
            left_frame,
            text="No Image Loaded\n\nClick 'Camera' to start live feed\nClick 'Upload' to select an image\nClick 'Test' to evaluate test sample",
            font=("Segoe UI", 12),
            bg="#0F172A",
            fg="#94A3B8",
            justify=tk.CENTER
        )
        self.screen_lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Mode Indicator beneath screen
        self.mode_lbl = tk.Label(
            left_frame,
            text="Mode: Idle",
            font=("Segoe UI", 10, "italic"),
            bg="#FFFFFF",
            fg="#64748B"
        )
        self.mode_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 6))

        # ----------------------------------------------------------------------
        # RIGHT COLUMN: 4 Action Buttons & Classification Results
        # ----------------------------------------------------------------------
        right_frame = tk.Frame(main_frame, bg="#F4F6F9", width=360)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_frame.pack_propagate(False)

        # --- Section A: 4 Action Buttons ---
        btn_frame = tk.LabelFrame(
            right_frame,
            text=" Control Actions ",
            font=("Segoe UI", 11, "bold"),
            bg="#FFFFFF",
            fg="#334155",
            bd=2,
            relief=tk.GROOVE
        )
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        # Button Style Helper
        btn_font = ("Segoe UI", 11, "bold")

        # 1. Camera Button
        self.btn_camera = tk.Button(
            btn_frame,
            text="📷 Start Camera",
            font=btn_font,
            bg="#0284C7",
            fg="#FFFFFF",
            activebackground="#0369A1",
            activeforeground="#FFFFFF",
            cursor="hand2",
            relief=tk.FLAT,
            height=2,
            command=self.toggle_camera
        )
        self.btn_camera.pack(fill=tk.X, padx=12, pady=5)

        # 2. Capture Button
        self.btn_capture = tk.Button(
            btn_frame,
            text="📸 Capture & Classify",
            font=btn_font,
            bg="#4F46E5",
            fg="#FFFFFF",
            activebackground="#4338CA",
            activeforeground="#FFFFFF",
            cursor="hand2",
            relief=tk.FLAT,
            height=2,
            command=self.capture_frame
        )
        self.btn_capture.pack(fill=tk.X, padx=12, pady=5)

        # 3. Upload Button
        self.btn_upload = tk.Button(
            btn_frame,
            text="📁 Upload Image",
            font=btn_font,
            bg="#0D9488",
            fg="#FFFFFF",
            activebackground="#0F766E",
            activeforeground="#FFFFFF",
            cursor="hand2",
            relief=tk.FLAT,
            height=2,
            command=self.upload_image
        )
        self.btn_upload.pack(fill=tk.X, padx=12, pady=5)

        # 4. Test Button
        self.btn_test = tk.Button(
            btn_frame,
            text="🧪 Test Sample",
            font=btn_font,
            bg="#D97706",
            fg="#FFFFFF",
            activebackground="#B45309",
            activeforeground="#FFFFFF",
            cursor="hand2",
            relief=tk.FLAT,
            height=2,
            command=self.test_sample
        )
        self.btn_test.pack(fill=tk.X, padx=12, pady=(5, 10))

        # --- Section B: Prediction Results Card ---
        results_frame = tk.LabelFrame(
            right_frame,
            text=" Classification Result ",
            font=("Segoe UI", 11, "bold"),
            bg="#FFFFFF",
            fg="#334155",
            bd=2,
            relief=tk.GROOVE
        )
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Detected Class Badge
        self.class_badge_lbl = tk.Label(
            results_frame,
            text="NO DETECTION",
            font=("Segoe UI", 16, "bold"),
            bg="#E2E8F0",
            fg="#475569",
            padx=10,
            pady=8
        )
        self.class_badge_lbl.pack(fill=tk.X, padx=12, pady=(10, 5))

        # Confidence & Detection Count
        stats_frame = tk.Frame(results_frame, bg="#FFFFFF")
        stats_frame.pack(fill=tk.X, padx=12, pady=3)

        self.conf_lbl = tk.Label(
            stats_frame,
            text="Confidence: --",
            font=("Segoe UI", 10, "bold"),
            bg="#FFFFFF",
            fg="#1E293B"
        )
        self.conf_lbl.pack(side=tk.LEFT)

        self.count_lbl = tk.Label(
            stats_frame,
            text="Objects: 0",
            font=("Segoe UI", 10),
            bg="#FFFFFF",
            fg="#64748B"
        )
        self.count_lbl.pack(side=tk.RIGHT)

        # Inference Time
        self.speed_lbl = tk.Label(
            results_frame,
            text="Inference Time: -- ms",
            font=("Segoe UI", 9),
            bg="#FFFFFF",
            fg="#94A3B8"
        )
        self.speed_lbl.pack(anchor=tk.W, padx=12, pady=(2, 6))

        # Breakdown Listbox (for multiple detected objects)
        lbl_breakdown = tk.Label(
            results_frame,
            text="Detected Objects:",
            font=("Segoe UI", 9, "bold"),
            bg="#FFFFFF",
            fg="#334155"
        )
        lbl_breakdown.pack(anchor=tk.W, padx=12, pady=(4, 0))

        self.items_listbox = tk.Listbox(
            results_frame,
            height=3,
            font=("Segoe UI", 9),
            bg="#F8FAFC",
            fg="#1E293B",
            bd=1,
            relief=tk.SOLID
        )
        self.items_listbox.pack(fill=tk.X, padx=12, pady=(2, 6))

        # Disposal / Project Application Guidance
        lbl_guidance = tk.Label(
            results_frame,
            text="Eco-Disposal Guidelines:",
            font=("Segoe UI", 9, "bold"),
            bg="#FFFFFF",
            fg="#0F766E"
        )
        lbl_guidance.pack(anchor=tk.W, padx=12, pady=(4, 0))

        self.guidelines_txt = tk.Text(
            results_frame,
            height=6,
            font=("Segoe UI", 9),
            bg="#F0FDF4",
            fg="#166534",
            wrap=tk.WORD,
            bd=1,
            relief=tk.SOLID
        )
        self.guidelines_txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=(2, 10))
        self.guidelines_txt.insert(tk.END, DISPOSAL_GUIDELINES["None"])
        self.guidelines_txt.config(state=tk.DISABLED)

    # --------------------------------------------------------------------------
    # 3. Camera Controls & Live Feed Loop
    # --------------------------------------------------------------------------
    def toggle_camera(self):
        """Starts or stops the live webcam feed."""
        if not self.is_camera_running:
            # Try opening the default webcam (index 0)
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not access webcam (index 0).\nPlease verify camera connection or permissions.")
                return

            self.is_camera_running = True
            self.btn_camera.config(text="⏹ Stop Camera", bg="#DC2626")
            self.mode_lbl.config(text="Mode: Live Camera Feed Active")
            self.update_webcam()
        else:
            self.stop_camera()

    def stop_camera(self):
        """Releases the webcam safely."""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_camera.config(text="📷 Start Camera", bg="#0284C7")
        self.mode_lbl.config(text="Mode: Camera Stopped")

    def update_webcam(self):
        """Tkinter update loop for smooth live webcam streaming."""
        if self.is_camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                # Display current frame on the left screen
                self.display_image_on_screen(frame)

            # Schedule next frame in 30ms (~33 FPS)
            self.root.after(30, self.update_webcam)

    # --------------------------------------------------------------------------
    # 4. Button Action: Capture
    # --------------------------------------------------------------------------
    def capture_frame(self):
        """Captures current camera frame, freezes it, and runs YOLO inference."""
        if not self.is_camera_running or self.current_frame is None:
            messagebox.showinfo("Camera Inactive", "Please click 'Start Camera' first before capturing a frame.")
            return

        captured_img = self.current_frame.copy()
        # Pause camera live feed so user can examine results
        self.stop_camera()
        self.mode_lbl.config(text="Mode: Captured Image from Webcam")

        # Run detection
        self.run_inference_and_display(captured_img)

    # --------------------------------------------------------------------------
    # 5. Button Action: Upload Image
    # --------------------------------------------------------------------------
    def upload_image(self):
        """Opens file dialog, loads an image, and runs inference."""
        if self.is_camera_running:
            self.stop_camera()

        file_path = filedialog.askopenfilename(
            title="Select Waste Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.PNG")]
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image from:\n{file_path}")
            return

        self.current_frame = img.copy()
        self.mode_lbl.config(text=f"Mode: Uploaded Image ({os.path.basename(file_path)})")
        self.run_inference_and_display(img)

    # --------------------------------------------------------------------------
    # 6. Button Action: Test Sample
    # --------------------------------------------------------------------------
    def test_sample(self):
        """Picks a random test image from the dataset and demonstrates inference."""
        if self.is_camera_running:
            self.stop_camera()

        test_dir = os.path.join("Dataset", "Final_merged_dataset", "test", "images")
        if not os.path.exists(test_dir):
            test_dir = os.path.join("merged_dataset", "test", "images")
        if not os.path.exists(test_dir):
            # Fallback: check original dataset test folders
            fallback_dirs = [
                os.path.join("Dataset", d, "test", "images")
                for d in os.listdir("Dataset") if os.path.isdir(os.path.join("Dataset", d))
            ]
            valid_dirs = [d for d in fallback_dirs if os.path.exists(d)]
            if valid_dirs:
                test_dir = valid_dirs[0]
            else:
                messagebox.showinfo("Dataset Not Found", "No test images folder found.\nPlease run 'python merge_datasets.py' first.")
                return

        images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not images:
            messagebox.showinfo("No Images", f"No test images found in {test_dir}")
            return

        selected = random.choice(images)
        full_path = os.path.join(test_dir, selected)
        img = cv2.imread(full_path)
        if img is None:
            messagebox.showerror("Error", f"Could not read test image: {selected}")
            return

        self.current_frame = img.copy()
        self.mode_lbl.config(text=f"Mode: Test Sample Image ({selected})")
        self.run_inference_and_display(img)

    # --------------------------------------------------------------------------
    # 7. YOLOv11n Inference & Visual Overlay
    # --------------------------------------------------------------------------
    def run_inference_and_display(self, bgr_image):
        """Runs YOLOv11n inference on image, draws boxes, and updates display/panel."""
        if self.model is None:
            messagebox.showerror("Model Missing", "YOLO model is not initialized.")
            return

        start_time = time.time()

        # Run Ultralytics inference
        # conf=0.25: confidence threshold
        results = self.model.predict(bgr_image, conf=0.25, verbose=False)
        inference_time_ms = (time.time() - start_time) * 1000

        annotated_frame = bgr_image.copy()
        detected_objects = []

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                # Get predicted class name
                if hasattr(self.model, "names") and cls_id in self.model.names:
                    cls_name = self.model.names[cls_id]
                elif cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                else:
                    cls_name = f"Class_{cls_id}"

                detected_objects.append((cls_name, conf))

                # Draw bounding box & label on image using class color
                color_bgr = CLASS_COLORS.get(cls_name, CLASS_COLORS["Unknown"])["bgr"]
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 3)

                label_text = f"{cls_name} {conf * 100:.1f}%"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - 25)), (x1 + w, max(0, y1)), color_bgr, -1)
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        self.last_annotated_frame = annotated_frame
        self.display_image_on_screen(annotated_frame)
        self.update_results_panel(detected_objects, inference_time_ms)

    # --------------------------------------------------------------------------
    # 8. Render Frame on Left Screen
    # --------------------------------------------------------------------------
    def display_image_on_screen(self, bgr_image):
        """Scales image proportionally to fit screen label and updates PhotoImage."""
        screen_w = max(100, self.screen_lbl.winfo_width())
        screen_h = max(100, self.screen_lbl.winfo_height())

        # Fallback to default dimensions if widget hasn't rendered yet
        if screen_w <= 1 or screen_h <= 1:
            screen_w, screen_h = 700, 520

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Resize proportionally to fit display screen
        pil_img.thumbnail((screen_w - 20, screen_h - 20), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(pil_img)

        # Keep reference to avoid garbage collection
        self.screen_lbl.config(image=photo_img, text="")
        self.screen_lbl.image = photo_img

    # --------------------------------------------------------------------------
    # 9. Update Results & Eco-Disposal Panel
    # --------------------------------------------------------------------------
    def update_results_panel(self, detected_objects, inference_time_ms):
        """Updates right-side information card with prediction details."""
        self.speed_lbl.config(text=f"Inference Time: {inference_time_ms:.1f} ms")
        self.items_listbox.delete(0, tk.END)

        if not detected_objects:
            self.class_badge_lbl.config(
                text="NO WASTE DETECTED",
                bg="#E2E8F0",
                fg="#64748B"
            )
            self.conf_lbl.config(text="Confidence: --")
            self.count_lbl.config(text="Objects: 0")
            self.items_listbox.insert(tk.END, "No bounding box found.")
            self.show_guidelines("None")
            return

        # Sort detections by confidence descending
        detected_objects.sort(key=lambda x: x[1], reverse=True)
        top_class, top_conf = detected_objects[0]

        # Update primary badge
        badge_bg = CLASS_COLORS.get(top_class, CLASS_COLORS["Unknown"])["hex"]
        self.class_badge_lbl.config(
            text=f"{top_class.upper().replace('_', ' ')}",
            bg=badge_bg,
            fg="#FFFFFF"
        )
        self.conf_lbl.config(text=f"Confidence: {top_conf * 100:.1f}%")
        self.count_lbl.config(text=f"Objects: {len(detected_objects)}")

        # Populate breakdown list
        for cls, conf in detected_objects:
            self.items_listbox.insert(tk.END, f"• {cls}: {conf * 100:.1f}%")

        # Update guidelines
        self.show_guidelines(top_class)

    def show_guidelines(self, waste_class):
        """Displays domain guidelines for the classified waste."""
        guide_text = DISPOSAL_GUIDELINES.get(waste_class, DISPOSAL_GUIDELINES["None"])
        self.guidelines_txt.config(state=tk.NORMAL)
        self.guidelines_txt.delete("1.0", tk.END)
        self.guidelines_txt.insert(tk.END, guide_text)
        self.guidelines_txt.config(state=tk.DISABLED)

    # --------------------------------------------------------------------------
    # 10. Clean Exit
    # --------------------------------------------------------------------------
    def on_closing(self):
        """Ensures camera is properly closed when window is dismissed."""
        self.stop_camera()
        self.root.destroy()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartWasteApp(root)
    root.mainloop()
