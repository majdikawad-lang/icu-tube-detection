const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const statusPanel = document.getElementById('statusPanel');
const statusText = document.getElementById('statusText');
const confidenceVal = document.getElementById('confidenceVal');
const distanceVal = document.getElementById('distanceVal');
const resultImage = document.getElementById('resultImage');

let audioCtx = null;
let alarmInterval = null;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

function playBeep() {
    if (!audioCtx) return;
    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    
    osc.type = 'square';
    osc.frequency.setValueAtTime(800, audioCtx.currentTime);
    osc.frequency.setValueAtTime(1200, audioCtx.currentTime + 0.1);
    
    gainNode.gain.setValueAtTime(0.5, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
    
    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    
    osc.start();
    osc.stop(audioCtx.currentTime + 0.3);
}

function startAlarm() {
    initAudio();
    playBeep();
    alarmInterval = setInterval(() => {
        playBeep();
        setTimeout(playBeep, 150);
    }, 1000);
}

function stopAlarm() {
    if (alarmInterval) {
        clearInterval(alarmInterval);
        alarmInterval = null;
    }
}

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#38bdf8';
    dropzone.style.background = 'rgba(56, 189, 248, 0.05)';
});

dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'rgba(255, 255, 255, 0.2)';
    dropzone.style.background = 'rgba(255, 255, 255, 0.02)';
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    if(e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFile(fileInput.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if(e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.name.endsWith('.dcm')) {
        alert("Please upload a valid DICOM (.dcm) file.");
        return;
    }
    
    // User interacted, we can init audio
    initAudio();
    
    dropzone.classList.add('hidden');
    loader.classList.remove('hidden');
    results.classList.add('hidden');
    stopAlarm();
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loader.classList.add('hidden');
        showResults(data);
    })
    .catch(err => {
        alert('Error processing image: ' + err);
        loader.classList.add('hidden');
        dropzone.classList.remove('hidden');
    });
}

function showResults(data) {
    results.classList.remove('hidden');
    
    resultImage.src = `data:image/jpeg;base64,${data.image_base64}`;
    confidenceVal.innerText = `${(data.confidence * 100).toFixed(1)}%`;
    distanceVal.innerText = `${data.distance_px.toFixed(1)} px`;
    
    statusPanel.className = 'status-panel'; // reset
    if (data.status === 'Safe') {
        statusText.innerText = '✅ SAFE POSITION';
        statusPanel.classList.add('safe');
    } else {
        statusText.innerHTML = '🚨 TUBE<br>DISPLACED';
        statusPanel.classList.add('danger');
        startAlarm();
    }
    
    // Add reset button dynamically
    if (!document.getElementById('resetBtn')) {
        const btn = document.createElement('button');
        btn.id = 'resetBtn';
        btn.className = 'reset-btn';
        btn.innerText = 'Analyze Another Image';
        btn.onclick = () => {
            stopAlarm();
            results.classList.add('hidden');
            dropzone.classList.remove('hidden');
            dropzone.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            dropzone.style.background = 'rgba(255, 255, 255, 0.02)';
            fileInput.value = '';
        };
        statusPanel.appendChild(btn);
    }
}
