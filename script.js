document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const mainDisplayImg = document.getElementById('main-display-img');
    const imageTag = document.getElementById('image-tag');
    const loadingOverlay = document.getElementById('loading-overlay');
    const viewModeTag = document.getElementById('view-mode-tag');

    const cameraContainer = document.getElementById('camera-container');
    const webcamVideo = document.getElementById('webcam-video');
    const btnCameraCapture = document.getElementById('btn-camera-capture');
    const btnCameraStop = document.getElementById('btn-camera-stop');

    const btnTriggerUpload = document.getElementById('btn-trigger-upload');
    const btnTriggerCamera = document.getElementById('btn-trigger-camera');
    const btnTriggerSample = document.getElementById('btn-trigger-sample');
    const btnRunAnalysis = document.getElementById('btn-run-analysis');
    const btnResetAll = document.getElementById('btn-reset-all');

    const primaryClassBadge = document.getElementById('primary-class-badge');
    const primaryConfVal = document.getElementById('primary-conf-val');
    const objectCountVal = document.getElementById('object-count-val');
    const inferenceTimeVal = document.getElementById('inference-time-val');
    const progressPercentLabel = document.getElementById('progress-percent-label');
    const confidenceProgressFill = document.getElementById('confidence-progress-fill');
    const detectionsCounter = document.getElementById('detections-counter');
    const detectionsList = document.getElementById('detections-list');

    const guidelineAction = document.getElementById('guideline-action');
    const guidelineCaution = document.getElementById('guideline-caution');
    const guidelineBenefit = document.getElementById('guideline-benefit');

    const sampleButtons = document.querySelectorAll('.btn-sample');

    let currentImageFile = null;
    let currentImageBase64 = null;
    let cameraStream = null;
    let isCameraActive = false;

    // ------------------------------------------------------------
    // FILE UPLOAD
    // ------------------------------------------------------------

    btnTriggerUpload.addEventListener('click', () => {
        console.log('[UPLOAD] Upload button clicked');
        stopCamera();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        console.log('[UPLOAD] Drop zone clicked');
        stopCamera();
        fileInput.click();
    });

    fileInput.addEventListener('change', (event) => {
        console.log('[UPLOAD] File input changed');

        if (!event.target.files || !event.target.files.length) {
            console.warn('[UPLOAD] No file selected');
            return;
        }

        const file = event.target.files[0];

        console.log('[UPLOAD] Selected file:', file.name);
        console.log('[UPLOAD] File type:', file.type);
        console.log('[UPLOAD] File size:', file.size);

        handleSelectedFile(file);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, event => {
            event.preventDefault();
            event.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, event => {
            event.preventDefault();
            event.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', event => {
        const files = event.dataTransfer.files;

        if (files && files.length > 0) {
            console.log('[UPLOAD] Dropped file:', files[0].name);
            handleSelectedFile(files[0]);
        }
    });

    function handleSelectedFile(file) {
        const validTypes = [
            'image/jpeg',
            'image/png',
            'image/webp',
            'image/bmp'
        ];

        const validExtension =
            /\.(jpg|jpeg|png|webp|bmp)$/i.test(file.name);

        if (!validTypes.includes(file.type) && !validExtension) {
            alert('Please select a valid image file (JPG, PNG, WEBP, or BMP).');
            return;
        }

        currentImageFile = file;
        currentImageBase64 = null;

        console.log('[UPLOAD] File accepted:', file.name);

        const reader = new FileReader();

        reader.onload = event => {
            console.log('[UPLOAD] Image preview loaded');

            showImagePreview(
                event.target.result,
                `Mode: Uploaded Image (${file.name})`,
                'Raw Uploaded Image'
            );

            btnRunAnalysis.disabled = false;
        };

        reader.onerror = () => {
            console.error('[UPLOAD] FileReader error');
            alert('Could not read the selected image.');
        };

        reader.readAsDataURL(file);
    }

    // ------------------------------------------------------------
    // CAMERA
    // ------------------------------------------------------------

    btnTriggerCamera.addEventListener('click', async () => {
        if (isCameraActive) {
            stopCamera();
        } else {
            await startCamera();
        }
    });

    async function startCamera() {
        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {
            alert('Camera is not supported. Please use file upload.');
            return;
        }

        try {
            stopCamera();

            hideAllDisplayScreens();
            cameraContainer.classList.remove('hidden');

            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                },
                audio: false
            });

            webcamVideo.srcObject = cameraStream;
            await webcamVideo.play();

            isCameraActive = true;

            btnTriggerCamera.innerHTML =
                '<span class="btn-icon">⏹</span> Stop Camera';

            viewModeTag.textContent =
                'Mode: Live Camera Feed Active';

            btnRunAnalysis.disabled = true;

        } catch (error) {
            console.error('[CAMERA] Error:', error);
            alert('Could not access camera. Please use file upload.');
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

        if (btnTriggerCamera) {
            btnTriggerCamera.innerHTML =
                '<span class="btn-icon">📷</span> Use Camera';
        }

        if (cameraContainer) {
            cameraContainer.classList.add('hidden');
        }

        if (!currentImageFile && !currentImageBase64) {
            dropZone.classList.remove('hidden');
            viewModeTag.textContent = 'Mode: Idle';
        }
    }

    btnCameraStop.addEventListener('click', stopCamera);

    btnCameraCapture.addEventListener('click', () => {
        if (!isCameraActive || !webcamVideo) return;

        const canvas = document.createElement('canvas');

        canvas.width = webcamVideo.videoWidth || 640;
        canvas.height = webcamVideo.videoHeight || 480;

        const ctx = canvas.getContext('2d');

        ctx.drawImage(
            webcamVideo,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const capturedBase64 =
            canvas.toDataURL('image/jpeg', 0.90);

        stopCamera();

        currentImageFile = null;
        currentImageBase64 = capturedBase64;

        showImagePreview(
            capturedBase64,
            'Mode: Captured from Camera',
            'Webcam Captured Frame'
        );

        btnRunAnalysis.disabled = false;

        executePrediction();
    });

    // ------------------------------------------------------------
    // TEST SAMPLE
    // ------------------------------------------------------------

    btnTriggerSample.addEventListener('click', async () => {
        stopCamera();

        loadingOverlay.classList.remove('hidden');

        try {
            console.log('[SAMPLE] Requesting /api/sample');

            const response = await fetch('/api/sample');

            const contentType =
                response.headers.get('content-type') || '';

            if (!contentType.includes('application/json')) {
                const text = await response.text();

                throw new Error(
                    `Server returned ${response.status}: ${
                        text || 'Empty response'
                    }`
                );
            }

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(
                    data.error || 'Failed to retrieve sample.'
                );
            }

            currentImageFile = null;
            currentImageBase64 = data.image_base64;

            showImagePreview(
                data.image_base64,
                `Mode: Test Sample (${data.filename})`,
                'Test Sample Image'
            );

            btnRunAnalysis.disabled = false;

            executePrediction();

        } catch (error) {
            console.error('[SAMPLE] Error:', error);

            alert(
                `Could not load test sample: ${error.message}`
            );

        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    sampleButtons.forEach(button => {
        button.addEventListener('click', () => {
            btnTriggerSample.click();
        });
    });

    // ------------------------------------------------------------
    // YOLO PREDICTION
    // ------------------------------------------------------------

    btnRunAnalysis.addEventListener('click', () => {
        console.log('[PREDICT] Analyze button clicked');
        executePrediction();
    });

    async function executePrediction() {
        if (!currentImageFile && !currentImageBase64) {
            alert('Please upload or capture an image first.');
            return;
        }

        console.log('======================================');
        console.log('[PREDICT] Starting prediction');
        console.log('[PREDICT] Has file:', !!currentImageFile);
        console.log('[PREDICT] Has base64:', !!currentImageBase64);
        console.log('======================================');

        loadingOverlay.classList.remove('hidden');

        const loadingText =
            document.querySelector('.loading-text');

        if (loadingText) {
            loadingText.textContent =
                'Analyzing image with YOLOv11n...';
        }

        btnRunAnalysis.disabled = true;
        btnTriggerUpload.disabled = true;
        btnTriggerCamera.disabled = true;
        btnTriggerSample.disabled = true;

        try {
            let response;

            if (currentImageFile) {
                console.log('[PREDICT] Sending multipart image to /predict');

                const formData = new FormData();

                formData.append(
                    'image',
                    currentImageFile,
                    currentImageFile.name
                );

                response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

            } else {
                console.log('[PREDICT] Sending base64 image to /predict');

                response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image_base64: currentImageBase64
                    })
                });
            }

            console.log(
                '[PREDICT] Server response:',
                response.status,
                response.statusText
            );

            const contentType =
                response.headers.get('content-type') || '';

            if (!contentType.includes('application/json')) {
                const text = await response.text();

                throw new Error(
                    `Server returned ${response.status}. ${
                        text || 'Empty response from server.'
                    }`
                );
            }

            const data = await response.json();

            console.log('[PREDICT] Server data:', data);

            if (!response.ok || !data.success) {
                throw new Error(
                    data.error || 'Prediction request failed.'
                );
            }

            renderResults(data);

        } catch (error) {
            console.error('[PREDICT] ERROR:', error);

            alert(
                `Classification error: ${error.message}`
            );

        } finally {
            loadingOverlay.classList.add('hidden');

            btnRunAnalysis.disabled = false;
            btnTriggerUpload.disabled = false;
            btnTriggerCamera.disabled = false;
            btnTriggerSample.disabled = false;
        }
    }

    // ------------------------------------------------------------
    // RENDER RESULTS
    // ------------------------------------------------------------

    function renderResults(data) {
        console.log('[RESULT] Rendering results:', data);

        if (data.annotated_image) {
            mainDisplayImg.src = data.annotated_image;

            imageTag.textContent =
                'YOLOv11n Annotated Result';

            viewModeTag.textContent =
                'Mode: YOLOv11n Inference Complete';
        }

        const hasDetections =
            Number(data.object_count || 0) > 0;

        if (hasDetections) {
            primaryClassBadge.textContent =
                String(data.primary_class_display || 'UNKNOWN')
                    .toUpperCase();

            primaryClassBadge.style.backgroundColor =
                data.primary_color_hex || '#0284C7';

            primaryConfVal.textContent =
                `${Number(data.primary_confidence || 0).toFixed(1)}%`;

            objectCountVal.textContent =
                data.object_count;

            inferenceTimeVal.textContent =
                `${Number(data.inference_time_ms || 0).toFixed(1)} ms`;

            const conf = Math.min(
                100,
                Math.max(
                    0,
                    Number(data.primary_confidence || 0)
                )
            );

            confidenceProgressFill.style.width =
                `${conf}%`;

            progressPercentLabel.textContent =
                `${conf.toFixed(1)}%`;

        } else {
            primaryClassBadge.textContent =
                'NO WASTE DETECTED';

            primaryClassBadge.style.backgroundColor =
                '#64748B';

            primaryConfVal.textContent = '--';
            objectCountVal.textContent = '0';

            inferenceTimeVal.textContent =
                `${Number(data.inference_time_ms || 0).toFixed(1)} ms`;

            confidenceProgressFill.style.width = '0%';
            progressPercentLabel.textContent = '0%';
        }

        detectionsList.innerHTML = '';

        if (
            hasDetections &&
            Array.isArray(data.detections) &&
            data.detections.length > 0
        ) {
            detectionsCounter.textContent =
                `${data.detections.length} Items`;

            data.detections.forEach((item, index) => {
                const row =
                    document.createElement('div');

                row.className = 'detection-row';

                const color =
                    item.color_hex || '#0284C7';

                const confidence =
                    Number(item.confidence || 0);

                row.innerHTML = `
                    <div class="detection-left">
                        <span
                            class="detection-dot"
                            style="background-color:${color};">
                        </span>

                        <span class="detection-name">
                            ${index + 1}. ${item.class_name}
                        </span>
                    </div>

                    <span
                        class="detection-conf"
                        style="color:${color};">
                        ${confidence.toFixed(1)}%
                    </span>
                `;

                detectionsList.appendChild(row);
            });

        } else {
            detectionsCounter.textContent = '0 Items';

            detectionsList.innerHTML = `
                <div class="empty-list-placeholder">
                    No bounding box found.
                    Please try another image or a clearer view.
                </div>
            `;
        }

        if (data.guidelines) {
            guidelineAction.textContent =
                data.guidelines.action || '';

            guidelineCaution.textContent =
                data.guidelines.caution || '';

            guidelineBenefit.textContent =
                data.guidelines.benefit || '';
        }
    }

    // ------------------------------------------------------------
    // RESET
    // ------------------------------------------------------------

    btnResetAll.addEventListener('click', resetState);

    function resetState() {
        stopCamera();

        currentImageFile = null;
        currentImageBase64 = null;

        fileInput.value = '';

        hideAllDisplayScreens();

        dropZone.classList.remove('hidden');

        btnRunAnalysis.disabled = true;

        viewModeTag.textContent = 'Mode: Idle';

        primaryClassBadge.textContent =
            'AWAITING IMAGE';

        primaryClassBadge.style.backgroundColor =
            '#64748B';

        primaryConfVal.textContent = '--';
        objectCountVal.textContent = '0';
        inferenceTimeVal.textContent = '-- ms';

        confidenceProgressFill.style.width = '0%';
        progressPercentLabel.textContent = '0%';

        detectionsCounter.textContent = '0 Items';

        detectionsList.innerHTML = `
            <div class="empty-list-placeholder">
                No objects detected yet.
                Run analysis to view individual detections.
            </div>
        `;

        guidelineAction.textContent =
            'Load an image to receive domain-specific recycling instructions.';

        guidelineCaution.textContent =
            'Always wear proper protective gear when handling hazardous or unknown waste.';

        guidelineBenefit.textContent =
            'Responsible sorting prevents toxic landfill leaching and conserves raw materials.';
    }

    // ------------------------------------------------------------
    // DISPLAY HELPERS
    // ------------------------------------------------------------

    function hideAllDisplayScreens() {
        dropZone.classList.add('hidden');

        if (cameraContainer) {
            cameraContainer.classList.add('hidden');
        }

        if (previewContainer) {
            previewContainer.classList.add('hidden');
        }
    }

    function showImagePreview(srcUrl, modeText, tagText) {
        hideAllDisplayScreens();

        previewContainer.classList.remove('hidden');

        mainDisplayImg.src = srcUrl;

        imageTag.textContent = tagText;

        viewModeTag.textContent = modeText;
    }

    console.log(
        '[APP] Smart Waste Classification frontend loaded successfully.'
    );
});