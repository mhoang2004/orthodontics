// ===== CONFIG =====
const BACKEND_URL = localStorage.getItem('backendUrl') || 'http://localhost:8001';
const API_ENDPOINT = `${BACKEND_URL}/infer`;
const API_INTERACTIVE = `${BACKEND_URL}/infer-interactive`;

// ===== DOM ELEMENTS =====
const imageInput = document.getElementById('imageInput');
const uploadBox = document.getElementById('uploadBox');
const statusMessage = document.getElementById('statusMessage');
const loadingSpinner = document.getElementById('loadingSpinner');
const loadingText = document.getElementById('loadingText');
const comparisonSection = document.getElementById('comparisonSection');
const beforeImage = document.getElementById('beforeImage');
const afterImage = document.getElementById('afterImage');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const apiStatus = document.getElementById('apiStatus');

// Interactive Controls
const whitenessSlider = document.getElementById('whitenessSlider');
const alignmentSlider = document.getElementById('alignmentSlider');
const timestepsSlider = document.getElementById('timestepsSlider');
const whitenessValue = document.getElementById('whitenessValue');
const alignmentValue = document.getElementById('alignmentValue');
const timestepsValue = document.getElementById('timestepsValue');
const regenerateBtn = document.getElementById('regenerateBtn');
const resetParamsBtn = document.getElementById('resetParamsBtn');
const paramsChanged = document.getElementById('paramsChanged');

let currentFile = null;
let currentAfterImageBlob = null;
let isProcessing = false;

// Default param values
const DEFAULTS = {
    whiteness: 100,   // slider value (divide by 100 to get actual)
    alignment: 100,
    timesteps: 60,
};

// ===== EVENT LISTENERS =====
uploadBox.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', handleImageSelect);

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = 'var(--primary)';
    uploadBox.style.background = 'var(--primary-glow)';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '';
    uploadBox.style.background = '';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '';
    uploadBox.style.background = '';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleImageSelect();
    }
});

downloadBtn.addEventListener('click', downloadResult);
resetBtn.addEventListener('click', resetUI);

// Slider listeners
whitenessSlider.addEventListener('input', () => {
    whitenessValue.textContent = (whitenessSlider.value / 100).toFixed(2);
    checkParamsChanged();
});

alignmentSlider.addEventListener('input', () => {
    alignmentValue.textContent = (alignmentSlider.value / 100).toFixed(2);
    checkParamsChanged();
});

timestepsSlider.addEventListener('input', () => {
    timestepsValue.textContent = timestepsSlider.value;
    checkParamsChanged();
});

regenerateBtn.addEventListener('click', handleRegenerate);
resetParamsBtn.addEventListener('click', resetParams);

// ===== FUNCTIONS =====

function getSliderParams() {
    return {
        whiteness: parseFloat(whitenessSlider.value) / 100,
        alignment: parseFloat(alignmentSlider.value) / 100,
        timesteps: parseInt(timestepsSlider.value),
    };
}

function checkParamsChanged() {
    const w = parseInt(whitenessSlider.value);
    const a = parseInt(alignmentSlider.value);
    const t = parseInt(timestepsSlider.value);
    const changed = (w !== DEFAULTS.whiteness || a !== DEFAULTS.alignment || t !== DEFAULTS.timesteps);
    
    if (changed) {
        paramsChanged.classList.remove('hidden');
    } else {
        paramsChanged.classList.add('hidden');
    }
}

function resetParams() {
    whitenessSlider.value = DEFAULTS.whiteness;
    alignmentSlider.value = DEFAULTS.alignment;
    timestepsSlider.value = DEFAULTS.timesteps;
    whitenessValue.textContent = '1.00';
    alignmentValue.textContent = '1.00';
    timestepsValue.textContent = '60';
    paramsChanged.classList.add('hidden');
}

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
    showLoading(true, 'Đang xử lý lần đầu... Vui lòng chờ (1-5 phút)');
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
    if (isProcessing) return;
    isProcessing = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`${response.status}: ${errorText}`);
        }

        const blob = await response.blob();
        currentAfterImageBlob = blob;
        afterImage.src = URL.createObjectURL(blob);

        comparisonSection.classList.remove('hidden');
        showStatus('✓ Xử lý thành công! Dùng slider bên dưới để tinh chỉnh.', 'success');
        
        setTimeout(() => {
            comparisonSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
    } catch (error) {
        console.error('Processing error:', error);
        throw new Error(`Không thể xử lý ảnh: ${error.message}. Kiểm tra backend API?`);
    } finally {
        isProcessing = false;
    }
}

async function handleRegenerate() {
    if (!currentFile || isProcessing) return;
    
    isProcessing = true;
    regenerateBtn.disabled = true;
    
    const params = getSliderParams();
    showLoading(true, `Đang tái tạo... (whiteness=${params.whiteness.toFixed(2)}, alignment=${params.alignment.toFixed(2)}, steps=${params.timesteps})`);
    clearStatus();

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('whiteness', params.whiteness);
    formData.append('alignment', params.alignment);
    formData.append('timesteps', params.timesteps);

    try {
        const response = await fetch(API_INTERACTIVE, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`${response.status}: ${errorText}`);
        }

        const result = await response.json();

        if (result.status === 'success') {
            // Update after image from base64
            afterImage.src = result.image;
            
            // Convert base64 to blob for download
            const byteString = atob(result.image.split(',')[1]);
            const mimeString = result.image.split(',')[0].split(':')[1].split(';')[0];
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            currentAfterImageBlob = new Blob([ab], { type: mimeString });

            const p = result.params;
            showStatus(
                `✓ Tái tạo thành công! (whiteness=${p.whiteness.toFixed(2)}, alignment=${p.alignment.toFixed(2)}, steps=${p.timesteps})`,
                'success'
            );
            paramsChanged.classList.add('hidden');
        } else {
            throw new Error('Kết quả không hợp lệ');
        }
    } catch (error) {
        console.error('Regenerate error:', error);
        showStatus(`Lỗi tái tạo: ${error.message}`, 'error');
    } finally {
        isProcessing = false;
        regenerateBtn.disabled = false;
        showLoading(false);
    }
}

function downloadResult() {
    if (!currentAfterImageBlob) return;

    const url = URL.createObjectURL(currentAfterImageBlob);
    const link = document.createElement('a');
    link.href = url;
    
    const params = getSliderParams();
    const paramStr = `w${params.whiteness.toFixed(1)}_a${params.alignment.toFixed(1)}_t${params.timesteps}`;
    link.download = `tooth-alignment-${paramStr}-${Date.now()}.png`;
    
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
    resetParams();
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

function showLoading(show, text) {
    if (show) {
        loadingSpinner.classList.remove('hidden');
        if (text) loadingText.textContent = text;
    } else {
        loadingSpinner.classList.add('hidden');
    }
}

// ===== CHECK API STATUS =====
async function checkAPIStatus() {
    try {
        const response = await fetch(BACKEND_URL, { 
            method: 'GET',
            signal: AbortSignal.timeout(5000)
        });
        if (response.ok) {
            apiStatus.textContent = '✓ Kết nối';
            apiStatus.style.color = '#10B981';
        } else {
            throw new Error('Not OK');
        }
    } catch (error) {
        apiStatus.textContent = '✗ Không kết nối';
        apiStatus.style.color = '#EF4444';
        console.warn('Backend API not reachable:', error);
    }
}

// Check API on page load
window.addEventListener('load', checkAPIStatus);

// Log config
console.log(`%c🦷 Tooth Alignment AI`, 'font-size: 16px; font-weight: bold; color: #3B82F6;');
console.log(`Backend URL: ${BACKEND_URL}`);
console.log('To change: localStorage.setItem("backendUrl", "http://your-backend:port")');
