/**
 * Frontend Configuration for Smart Waste Classification
 * Handles Backend API URL for Localhost & Render Cloud Hosting
 */

// Default backend URL when deployed on Render
// Replace this with your actual Render service URL once deployed:
const DEFAULT_RENDER_BACKEND = "https://smart-waste-classification.onrender.com";

// Determine active backend URL (allows localStorage override for testing)
function getBackendUrl() {
    const savedUrl = localStorage.getItem("SWC_BACKEND_URL");
    if (savedUrl && savedUrl.trim() !== "") {
        return savedUrl.trim().replace(/\/+$/, "");
    }
    
    // If running locally in browser
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.protocol === "file:") {
        return "http://127.0.0.1:5000";
    }

    return DEFAULT_RENDER_BACKEND;
}

function setBackendUrl(url) {
    if (url && url.trim() !== "") {
        localStorage.setItem("SWC_BACKEND_URL", url.trim().replace(/\/+$/, ""));
    } else {
        localStorage.removeItem("SWC_BACKEND_URL");
    }
}
