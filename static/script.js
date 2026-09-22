/**
 * Smart Waste Classification System - Frontend Client Logic
 * Subject: IPA (Image Processing and Applications)
 * Model: YOLOv11n (Ultralytics)
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const mainDisplayImg = document.getElementById('main-display-img');
    const imageTag = document.getElementById('image-tag');
    const loadingOverlay = document.getElementById('loading-overlay');
    const viewModeTag = document.getElementById('view-mode-tag');

    // Camera Elements
    const cameraContainer = document.getElementById('camera-container');
    const webcamVideo = document.getElementById('webcam-video');
    const btnCameraCapture = document.getElementById('btn-camera-capture');
    const btnCameraStop = document.getElementById('btn-camera-stop');

    // Control Buttons
    const btnTriggerUpload = document.getElementById('btn-trigger-upload');
    const btnTriggerCamera = document.getElementById('btn-trigger-camera');
    const btnTriggerSample = document.getElementById('btn-trigger-sample');
    const btnRunAnalysis = document.getElementById('btn-run-analysis');
    const btnResetAll = document.getElementById('btn-reset-all');

    // Result Elements
    const primaryClassBadge = document.getElementById('primary-class-badge');
    const primaryConfVal = document.getElementById('primary-conf-val');
    const objectCountVal = document.getElementById('object-count-val');
    const inferenceTimeVal = document.getElementById('inference-time-val');
    const progressPercentLabel = document.getElementById('progress-percent-label');
    const confidenceProgressFill = document.getElementById('confidence-progress-fill');
    const detectionsCounter = document.getElementById('detections-counter');
    const detectionsList = document.getElementById('detections-list');

    // Guidelines Elements
    const guidelineAction = document.getElementById('guideline-action');
    const guidelineCaution = document.getElementById('guideline-caution');
    const guidelineBenefit = document.getElementById('guideline-benefit');

    // Quick Sample Buttons
    const sampleButtons = document.querySelectorAll('.btn-sample');

    // --- State Variables ---
    let currentImageFile = null;
    let currentImageBase64 = null;
    let cameraStream = null;
    let isCameraActive = false;

    // --------------------------------------------------------------------------
    // 1. File Upload & Drag-and-Drop
    // --------------------------------------------------------------------------
    btnTriggerUpload.addEventListener('click', () => {
        stopCamera();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        stopCamera();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag and drop handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    function handleSelectedFile(file) {
        // Validate file type
        const validExtensions = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
        if (!validExtensions.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i)) {
            alert('Please select a valid image file (JPG, PNG, WEBP, or BMP).');
            return;
        }

        currentImageFile = file;
        currentImageBase64 = null;

        const reader = new FileReader();
        reader.onload = (event) => {
            showImagePreview(event.target.result, `Mode: Uploaded Image (${file.name})`, 'Raw Uploaded Image');
            btnRunAnalysis.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // --------------------------------------------------------------------------
    // 2. Camera Controls (Webcam API via navigator.mediaDevices)
    // --------------------------------------------------------------------------
    btnTriggerCamera.addEventListener('click', async () => {
        if (isCameraActive) {
            stopCamera();
        } else {
            await startCamera();
        }
    });

    async function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Your browser does not support camera access or it is disabled in this context. Please use file upload.');
            return;
        }

        try {
            stopCamera();
            hideAllDisplayScreens();
            cameraContainer.classList.remove('hidden');

            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                },
                audio: false
            };

            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            webcamVideo.srcObject = cameraStream;
            await webcamVideo.play();

            isCameraActive = true;
            btnTriggerCamera.innerHTML = '<span class="btn-icon">⏹</span> Stop Camera';
            btnTriggerCamera.style.backgroundColor = '#DC2626';
            viewModeTag.textContent = 'Mode: Live Camera Feed Active';
            btnRunAnalysis.disabled = true;

        } catch (err) {
            console.error('Camera access error:', err);
            cameraContainer.classList.add('hidden');
            dropZone.classList.remove('hidden');
            let msg = 'Could not access camera.';
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                msg = 'Camera permission was denied. Please allow camera permissions in your browser or use file upload.';
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                msg = 'No camera hardware was detected on this device.';
            }
            alert(msg);
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        if (webcamVideo) {
            webcamVideo.srcObject = null;
        }
        isCameraActive = false;
        btnTriggerCamera.innerHTML = '<span class="btn-icon">📷</span> Use Camera';
        btnTriggerCamera.style.backgroundColor = '';
        if (cameraContainer && !cameraContainer.classList.contains('hidden')) {
            cameraContainer.classList.add('hidden');
            if (!currentImageFile && !currentImageBase64) {
                dropZone.classList.remove('hidden');
                viewModeTag.textContent = 'Mode: Idle';
            }
        }
    }

    btnCameraStop.addEventListener('click', () => {
        stopCamera();
    });

    btnCameraCapture.addEventListener('click', () => {
        if (!isCameraActive || !webcamVideo) return;

        // Draw current video frame to an in-memory canvas
        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth || 640;
        canvas.height = webcamVideo.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

        const capturedBase64 = canvas.toDataURL('image/jpeg', 0.92);
        stopCamera();

        currentImageFile = null;
        currentImageBase64 = capturedBase64;

        showImagePreview(capturedBase64, 'Mode: Captured from Camera', 'Webcam Captured Frame');
        btnRunAnalysis.disabled = false;

        // Automatically trigger YOLO classification on captured frame
        executePrediction();
    });

    // --------------------------------------------------------------------------
    // 3. Test Sample Feature (Mirroring Tkinter "Test Sample" functionality)
    // --------------------------------------------------------------------------
    btnTriggerSample.addEventListener('click', async () => {
        stopCamera();
        loadingOverlay.classList.remove('hidden');
        document.querySelector('.loading-text').textContent = 'Loading random test dataset sample...';

        try {
            const resp = await fetch('/api/sample');
            const data = await resp.json();
            loadingOverlay.classList.add('hidden');

            if (!resp.ok || !data.success) {
                alert(data.error || 'Failed to retrieve test dataset sample.');
                return;
            }

            currentImageFile = null;
            currentImageBase64 = data.image_base64;

            showImagePreview(data.image_base64, `Mode: Test Sample (${data.filename})`, 'Test Sample Image');
            btnRunAnalysis.disabled = false;

            // Automatically run analysis on the loaded sample
            executePrediction();

        } catch (err) {
            loadingOverlay.classList.add('hidden');
            console.error('Sample fetch error:', err);
            alert('Could not load test sample. Please check if dataset test images exist.');
        }
    });

    // Quick sample tags
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            btnTriggerSample.click();
        });
    });

    // --------------------------------------------------------------------------
    // 4. YOLOv11n Inference Execution
    // --------------------------------------------------------------------------
    btnRunAnalysis.addEventListener('click', () => {
        executePrediction();
    });

    async function executePrediction() {
        if (!currentImageFile && !currentImageBase64) {
            alert('Please upload or capture an image first.');
            return;
        }

        // Show loading state
        loadingOverlay.classList.remove('hidden');
        document.querySelector('.loading-text').textContent = 'Analyzing image with YOLOv11n...';
        btnRunAnalysis.disabled = true;
        btnTriggerUpload.disabled = true;
        btnTriggerCamera.disabled = true;
        btnTriggerSample.disabled = true;

        try {
            let response;

            if (currentImageFile) {
                const formData = new FormData();
                formData.append('image', currentImageFile);
                response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
            } else if (currentImageBase64) {
                response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_base64: currentImageBase64 })
                });
            }

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Prediction request failed.');
            }

            // Render YOLO Detection Results
            renderResults(data);

        } catch (err) {
            console.error('Prediction error:', err);
            alert(`Classification error: ${err.message}`);
        } finally {
            loadingOverlay.classList.add('hidden');
            btnRunAnalysis.disabled = false;
            btnTriggerUpload.disabled = false;
            btnTriggerCamera.disabled = false;
            btnTriggerSample.disabled = false;
        }
    }

    // --------------------------------------------------------------------------
    // 5. Render Results to Dashboard
    // --------------------------------------------------------------------------
    function renderResults(data) {
        // 1. Update Display Screen with Annotated YOLO Image
        if (data.annotated_image) {
            mainDisplayImg.src = data.annotated_image;
            imageTag.textContent = 'YOLOv11n Annotated Result';
            viewModeTag.textContent = 'Mode: YOLOv11n Inference Complete';
        }

        // 2. Primary Detection Card
        const hasDetections = data.object_count > 0;

        if (hasDetections) {
            primaryClassBadge.textContent = data.primary_class_display.toUpperCase();
            primaryClassBadge.style.backgroundColor = data.primary_color_hex || '#0284C7';
            primaryConfVal.textContent = `${data.primary_confidence.toFixed(1)}%`;
            objectCountVal.textContent = data.object_count;
            inferenceTimeVal.textContent = `${data.inference_time_ms.toFixed(1)} ms`;

            // Progress Bar
            const conf = Math.min(100, Math.max(0, data.primary_confidence));
            confidenceProgressFill.style.width = `${conf}%`;
            confidenceProgressFill.style.backgroundColor = data.primary_color_hex || '#0284C7';
            progressPercentLabel.textContent = `${conf.toFixed(1)}%`;

        } else {
            // No Detections State
            primaryClassBadge.textContent = 'NO WASTE DETECTED';
            primaryClassBadge.style.backgroundColor = '#64748B';
            primaryConfVal.textContent = '--';
            objectCountVal.textContent = '0';
            inferenceTimeVal.textContent = `${data.inference_time_ms.toFixed(1)} ms`;

            confidenceProgressFill.style.width = '0%';
            progressPercentLabel.textContent = '0%';
        }

        // 3. Individual Detections Breakdown List
        detectionsList.innerHTML = '';
        if (hasDetections && data.detections && data.detections.length > 0) {
            detectionsCounter.textContent = `${data.detections.length} Items`;
            data.detections.forEach((item, index) => {
                const row = document.createElement('div');
                row.className = 'detection-row';
                row.innerHTML = `
                    <div class="detection-left">
                        <span class="detection-dot" style="background-color: ${item.color_hex};"></span>
                        <span class="detection-name">${index + 1}. ${item.class_name}</span>
                    </div>
                    <span class="detection-conf" style="color: ${item.color_hex};">${item.confidence.toFixed(1)}%</span>
                `;
                detectionsList.appendChild(row);
            });
        } else {
            detectionsCounter.textContent = '0 Items';
            detectionsList.innerHTML = `
                <div class="empty-list-placeholder">
                    No bounding box found. Please try another image or a clearer view.
                </div>
            `;
        }

        // 4. Eco-Disposal Guidelines
        if (data.guidelines) {
            guidelineAction.textContent = data.guidelines.action;
            guidelineCaution.textContent = data.guidelines.caution;
            guidelineBenefit.textContent = data.guidelines.benefit;
        }
    }

    // --------------------------------------------------------------------------
    // 6. Reset & Clear Functionality
    // --------------------------------------------------------------------------
    btnResetAll.addEventListener('click', () => {
        resetState();
    });

    function resetState() {
        stopCamera();

        currentImageFile = null;
        currentImageBase64 = null;
        fileInput.value = '';

        hideAllDisplayScreens();
        dropZone.classList.remove('hidden');

        btnRunAnalysis.disabled = true;
        viewModeTag.textContent = 'Mode: Idle';

        // Reset Primary Card
        primaryClassBadge.textContent = 'AWAITING IMAGE';
        primaryClassBadge.style.backgroundColor = '#64748B';
        primaryConfVal.textContent = '--';
        objectCountVal.textContent = '0';
        inferenceTimeVal.textContent = '-- ms';
        confidenceProgressFill.style.width = '0%';
        progressPercentLabel.textContent = '0%';

        // Reset Detections List
        detectionsCounter.textContent = '0 Items';
        detectionsList.innerHTML = `
            <div class="empty-list-placeholder">
                No objects detected yet. Run analysis to view individual detections.
            </div>
        `;

        // Reset Guidelines
        guidelineAction.textContent = 'Load an image to receive domain-specific recycling instructions.';
        guidelineCaution.textContent = 'Always wear proper protective gear when handling hazardous or unknown waste.';
        guidelineBenefit.textContent = 'Responsible sorting prevents toxic landfill leaching and conserves raw materials.';
    }

    // --------------------------------------------------------------------------
    // Screen Display Helpers
    // --------------------------------------------------------------------------
    function hideAllDisplayScreens() {
        dropZone.classList.add('hidden');
        if (cameraContainer) cameraContainer.classList.add('hidden');
        if (previewContainer) previewContainer.classList.add('hidden');
    }

    function showImagePreview(srcUrl, modeText, tagText) {
        hideAllDisplayScreens();
        previewContainer.classList.remove('hidden');
        mainDisplayImg.src = srcUrl;
        imageTag.textContent = tagText;
        viewModeTag.textContent = modeText;
    }
});
