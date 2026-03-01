// ===== CONFIG =====
const BACKEND_URL = localStorage.getItem('backendUrl') || 'http://localhost:8001';
const API_ENDPOINT = `${BACKEND_URL}/infer`;

// ===== DOM ELEMENTS =====
const imageInput = document.getElementById('imageInput');
const uploadBox = document.querySelector('.upload-box');
const statusMessage = document.getElementById('statusMessage');
const loadingSpinner = document.getElementById('loadingSpinner');
const comparisonSection = document.getElementById('comparisonSection');
const beforeImage = document.getElementById('beforeImage');
const afterImage = document.getElementById('afterImage');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const apiStatus = document.getElementById('apiStatus');

let currentFile = null;
let currentAfterImageBlob = null;

// ===== EVENT LISTENERS =====
uploadBox.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', handleImageSelect);

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.backgroundColor = '#f0f6ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.backgroundColor = 'var(--surface)';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.backgroundColor = 'var(--surface)';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleImageSelect();
    }
});

downloadBtn.addEventListener('click', downloadResult);
resetBtn.addEventListener('click', resetUI);

// ===== FUNCTIONS =====
async function handleImageSelect() {
    const file = imageInput.files[0];
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) {
        showStatus('Lỗi: Kích thước file vượt quá 10MB', 'error');
        return;
    }

    currentFile = file;
    beforeImage.src = URL.createObjectURL(file);
    
    clearStatus();
    showLoading(true);
    comparisonSection.classList.add('hidden');

    try {
        await processImage(file);
    } catch (error) {
        showStatus(`Lỗi: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function processImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData,
            // không set Content-Type, browser sẽ tự set multipart/form-data
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`${response.status}: ${errorText}`);
        }

        const blob = await response.blob();
        currentAfterImageBlob = blob;
        afterImage.src = URL.createObjectURL(blob);

        comparisonSection.classList.remove('hidden');
        showStatus('✓ Xử lý thành công!', 'success');
        // Scroll to result
        setTimeout(() => {
            comparisonSection.scrollIntoView({ behavior: 'smooth' });
        }, 300);
    } catch (error) {
        console.error('Processing error:', error);
        throw new Error(`Không thể xử lý ảnh: ${error.message}. Kiểm tra backend API có chạy không?`);
    }
}

function downloadResult() {
    if (!currentAfterImageBlob) return;

    const url = URL.createObjectURL(currentAfterImageBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `tooth-alignment-result-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function resetUI() {
    imageInput.value = '';
    currentFile = null;
    currentAfterImageBlob = null;
    beforeImage.src = '';
    afterImage.src = '';
    comparisonSection.classList.add('hidden');
    clearStatus();
    uploadBox.scrollIntoView({ behavior: 'smooth' });
}

function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
}

function clearStatus() {
    statusMessage.className = 'status-message hidden';
    statusMessage.textContent = '';
}

function showLoading(show) {
    if (show) {
        loadingSpinner.classList.remove('hidden');
    } else {
        loadingSpinner.classList.add('hidden');
    }
}

// ===== CHECK API STATUS =====
async function checkAPIStatus() {
    try {
        // Thử gọi một endpoint đơn giản để check xem API có chạy không
        // Vì không có endpoint root, ta chỉ show warning nếu fetch fail
        const response = await fetch(BACKEND_URL, { method: 'HEAD' });
        apiStatus.textContent = '✓ Kết nối';
        apiStatus.style.color = '#2ecc71';
    } catch (error) {
        apiStatus.textContent = '✗ Không kết nối';
        apiStatus.style.color = '#ff6b6b';
        console.warn('Backend API not reachable:', error);
    }
}

// Check API on page load
window.addEventListener('load', checkAPIStatus);

// Optional: Allow user to set custom backend URL
console.log(`Frontend initialized. Backend URL: ${BACKEND_URL}`);
console.log('To change backend URL, run: localStorage.setItem("backendUrl", "http://your-backend:port")');
