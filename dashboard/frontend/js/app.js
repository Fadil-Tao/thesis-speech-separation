/**
 * Speech Separation Dashboard - Main Application
 */

// State management
const state = {
    currentPage: 'evaluation',
    selectedModel: null,
    selectedDataset: null,
    selectedSample: null,
    models: [],
    datasets: [],
    evaluationResults: null
};

// API base URL
const API_BASE = 'http://localhost:5000/api';

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initTabs();
    initUploadArea();
    loadModels();
    loadDatasets();
});

// Navigation
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');
    const pageTitle = document.getElementById('page-title');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            
            // Update active nav
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Update active page
            pages.forEach(p => p.classList.remove('active'));
            document.getElementById(`${page}-page`).classList.add('active');
            
            // Update title
            const titles = {
                'evaluation': 'Model Evaluation',
                'dataset': 'Dataset Overview',
                'comparison': 'Model Comparison'
            };
            pageTitle.textContent = titles[page];
            
            state.currentPage = page;
        });
    });

    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', () => {
        loadModels();
        loadDatasets();
    });
}

// Tabs
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${target}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
}

// Upload area
function initUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--accent-primary)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border-color)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    if (!file.type.startsWith('audio/')) {
        alert('Please upload an audio file');
        return;
    }

    // Create temporary URL for the file
    const url = URL.createObjectURL(file);
    
    state.selectedSample = {
        id: file.name,
        mix: url,  // For API - will need to be handled differently for uploads
        audioUrl: url,  // For audio element playback
        isUpload: true,
        file: file
    };

    showSelectedAudio(file.name, url, url);
}

// Load models
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();
        state.models = data.models;

        const select = document.getElementById('model-select');
        select.innerHTML = '<option value="">Select model...</option>';

        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            if (!model.exists) {
                option.disabled = true;
                option.textContent += ' (not found)';
            }
            select.appendChild(option);
        });

        select.addEventListener('change', (e) => {
            const modelId = e.target.value;
            if (modelId) {
                const model = state.models.find(m => m.id === modelId);
                state.selectedModel = modelId;
                showModelInfo(model);
            } else {
                state.selectedModel = null;
                document.getElementById('model-info').style.display = 'none';
            }
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function showModelInfo(model) {
    const infoDiv = document.getElementById('model-info');
    document.getElementById('model-speakers').textContent = model.num_speakers;
    document.getElementById('model-status').textContent = model.exists ? 'Available' : 'Not Found';
    document.getElementById('model-status').style.color = model.exists ? 'var(--success)' : 'var(--error)';
    infoDiv.style.display = 'block';
}

// Load datasets
async function loadDatasets() {
    try {
        const response = await fetch(`${API_BASE}/datasets`);
        const data = await response.json();
        state.datasets = data.datasets;

        // Populate dataset selects
        const selects = ['dataset-select', 'browser-dataset-select', 'batch-dataset-select'];
        selects.forEach(selectId => {
            const select = document.getElementById(selectId);
            select.innerHTML = '<option value="">Select dataset...</option>';
            
            data.datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset.id;
                option.textContent = dataset.name;
                select.appendChild(option);
            });
        });

        // Add change listeners
        document.getElementById('dataset-select').addEventListener('change', (e) => {
            const datasetId = e.target.value;
            if (datasetId) {
                state.selectedDataset = datasetId;
                loadSamples(datasetId, document.getElementById('split-select').value);
            }
        });

        document.getElementById('split-select').addEventListener('change', (e) => {
            if (state.selectedDataset) {
                loadSamples(state.selectedDataset, e.target.value);
            }
        });

        // Browser selects
        document.getElementById('browser-dataset-select').addEventListener('change', (e) => {
            const datasetId = e.target.value;
            if (datasetId) {
                loadSamplesTable(datasetId, document.getElementById('browser-split-select').value);
            }
        });

        document.getElementById('browser-split-select').addEventListener('change', (e) => {
            const datasetId = document.getElementById('browser-dataset-select').value;
            if (datasetId) {
                loadSamplesTable(datasetId, e.target.value);
            }
        });

        // Update dataset stats
        updateDatasetStats(data.datasets);

        // Setup batch evaluation button
        setupBatchEvaluation();
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

// Setup batch evaluation
function setupBatchEvaluation() {
    const runBatchBtn = document.getElementById('run-batch-btn');
    if (runBatchBtn) {
        runBatchBtn.addEventListener('click', runBatchEvaluation);
    }

    const exportBatchBtn = document.getElementById('export-batch-btn');
    if (exportBatchBtn) {
        exportBatchBtn.addEventListener('click', exportBatchResults);
    }
}

// Run batch evaluation
async function runBatchEvaluation() {
    const modelId = state.selectedModel;
    const datasetId = document.getElementById('batch-dataset-select').value;
    const split = document.getElementById('batch-split-select').value;
    const numSamples = document.getElementById('batch-num-samples').value;

    if (!modelId) {
        alert('Please select a model first');
        return;
    }

    if (!datasetId) {
        alert('Please select a dataset');
        return;
    }

    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/evaluate-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: modelId,
                dataset: datasetId,
                split: split,
                num_samples: parseInt(numSamples)
            })
        });

        const data = await response.json();
        showLoading(false);

        if (data.success) {
            showBatchResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        showLoading(false);
        console.error('Error running batch evaluation:', error);
        alert('Error running batch evaluation: ' + error.message);
    }
}

// Show batch results
function showBatchResults(data) {
    const resultsSection = document.getElementById('batch-results-section');
    resultsSection.style.display = 'block';

    // Show summary metrics
    const summaryGrid = document.getElementById('batch-summary-metrics');
    summaryGrid.innerHTML = '';

    const avgMetrics = data.average_metrics;
    
    if (avgMetrics.avg_si_snr !== undefined) {
        summaryGrid.innerHTML += `
            <div class="metric-card">
                <div class="metric-label">Average SI-SNR</div>
                <div class="metric-value">${avgMetrics.avg_si_snr}<span class="metric-unit">dB</span></div>
                <div class="metric-subtext">Std: ${avgMetrics.std_si_snr} dB</div>
            </div>
        `;
    }

    if (avgMetrics.avg_stoi !== undefined) {
        summaryGrid.innerHTML += `
            <div class="metric-card">
                <div class="metric-label">Average STOI</div>
                <div class="metric-value">${avgMetrics.avg_stoi}</div>
                <div class="metric-subtext">Std: ${avgMetrics.std_stoi}</div>
            </div>
        `;
    }

    summaryGrid.innerHTML += `
        <div class="metric-card">
            <div class="metric-label">Samples Evaluated</div>
            <div class="metric-value">${data.num_evaluated}</div>
        </div>
    `;

    // Populate results table
    const tbody = document.querySelector('#batch-results-table tbody');
    tbody.innerHTML = '';

    data.results.forEach(result => {
        Object.entries(result.speakers).forEach(([speaker, metrics]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><code>${result.file_id}</code></td>
                <td>${speaker}</td>
                <td>${metrics.si_snr.toFixed(2)} dB</td>
                <td>${metrics.stoi !== null ? metrics.stoi.toFixed(3) : 'N/A'}</td>
            `;
            tbody.appendChild(row);
        });
    });

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Export batch results
function exportBatchResults() {
    const table = document.getElementById('batch-results-table');
    const rows = table.querySelectorAll('tbody tr');
    
    let csv = 'Sample ID,Speaker,SI-SNR (dB),STOI\n';
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        const values = Array.from(cells).map(cell => cell.textContent.replace(' dB', ''));
        csv += values.join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch-evaluation-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function updateDatasetStats(datasets) {
    const statsGrid = document.getElementById('dataset-stats');
    
    datasets.forEach(dataset => {
        const totalSamples = Object.values(dataset.splits).reduce((a, b) => a + b, 0);
        const info = dataset.info;
        
        const card = document.createElement('div');
        card.className = 'stat-card';
        card.innerHTML = `
            <div class="stat-value">${totalSamples.toLocaleString()}</div>
            <div class="stat-label">${dataset.name} Total Samples</div>
        `;
        statsGrid.appendChild(card);

        if (info && info.total) {
            const hoursCard = document.createElement('div');
            hoursCard.className = 'stat-card';
            hoursCard.innerHTML = `
                <div class="stat-value">${Math.round(info.total.duration_hours)}h</div>
                <div class="stat-label">Total Duration</div>
            `;
            statsGrid.appendChild(hoursCard);
        }
    });
}

// Load samples
async function loadSamples(datasetId, split) {
    try {
        const response = await fetch(`${API_BASE}/dataset/${datasetId}/samples?split=${split}&limit=12`);
        const data = await response.json();

        const grid = document.getElementById('samples-grid');
        grid.innerHTML = '';

        data.samples.forEach(sample => {
            const card = createSampleCard(sample);
            grid.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading samples:', error);
    }
}

function createSampleCard(sample) {
    const card = document.createElement('div');
    card.className = 'sample-card';
    card.dataset.id = sample.id;

    // Get audio URL for playback - convert absolute path to relative path for API
    let audioUrl;
    // Store the actual file path for API calls
    let filePath = sample.mix;
    
    if (sample.mix.startsWith('blob:')) {
        audioUrl = sample.mix;
    } else {
        // Extract relative path from project root
        const match = sample.mix.match(/dataset\/synthetic\/.*/);
        audioUrl = match ? `/api/audio/dataset/${match[0]}` : sample.mix;
    }

    card.innerHTML = `
        <div class="sample-card-header">
            <span class="sample-id">${sample.id}</span>
            <span class="sample-meta">5.0s</span>
        </div>
        <div class="sample-waveform">
            <canvas class="waveform-preview" data-src="${audioUrl}"></canvas>
        </div>
    `;

    card.addEventListener('click', () => {
        // Remove selected from all cards
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');

        // Store both the file path (for API) and audio URL (for playback)
        state.selectedSample = {
            ...sample,
            mix: filePath,  // Keep original file path for API
            audioUrl: audioUrl  // URL for audio element
        };
        showSelectedAudio(sample.id, audioUrl, filePath);
    });

    // Draw waveform preview
    setTimeout(() => drawWaveformPreview(card.querySelector('.waveform-preview'), filePath), 0);

    return card;
}

async function drawWaveformPreview(canvas, audioUrl) {
    try {
        const response = await fetch(`${API_BASE}/waveform`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_path: audioUrl, points: 100 })
        });
        const data = await response.json();

        if (data.waveform) {
            drawWaveform(canvas, data.waveform, '#666');
        }
    } catch (error) {
        console.error('Error drawing waveform:', error);
    }
}

function drawWaveform(canvas, data, color = '#0070f3') {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const centerY = height / 2;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = color;

    const barWidth = width / data.length;
    const barGap = 1;

    data.forEach((value, i) => {
        const barHeight = Math.abs(value) * (height / 2);
        const x = i * barWidth;
        const y = centerY - barHeight / 2;
        
        ctx.fillRect(x, y, barWidth - barGap, barHeight);
    });
}

// Show selected audio
function showSelectedAudio(filename, audioUrl, filePath) {
    const section = document.getElementById('selected-audio-section');
    section.style.display = 'block';

    document.getElementById('selected-filename').textContent = filename;

    const audio = document.getElementById('mixture-audio');
    audio.src = audioUrl;

    // Draw waveform using file path
    const canvas = document.getElementById('mixture-waveform');
    drawWaveformPreview(canvas, filePath || audioUrl);

    // Setup evaluate button
    document.getElementById('evaluate-btn').onclick = () => runEvaluation();
}

// Run evaluation
async function runEvaluation() {
    if (!state.selectedModel || !state.selectedSample) {
        alert('Please select a model and audio sample');
        return;
    }

    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: state.selectedModel,
                audio_path: state.selectedSample.mix
            })
        });

        const data = await response.json();
        showLoading(false);

        if (data.success) {
            state.evaluationResults = data;
            showResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        showLoading(false);
        console.error('Error running evaluation:', error);
        alert('Error running evaluation: ' + error.message);
    }
}

function showResults(data) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';

    // Show metrics
    const metricsGrid = document.getElementById('metrics-grid');
    metricsGrid.innerHTML = '';

    if (data.metrics && Object.keys(data.metrics).length > 0) {
        // Calculate average metrics
        let totalSiSnr = 0;
        let totalStoi = 0;
        let count = 0;

        Object.values(data.metrics).forEach(m => {
            if (m.si_snr) {
                totalSiSnr += m.si_snr;
                count++;
            }
            if (m.stoi) {
                totalStoi += m.stoi;
            }
        });

        if (count > 0) {
            const avgSiSnr = totalSiSnr / count;
            const avgStoi = count > 0 ? totalStoi / count : 0;

            metricsGrid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-label">Average SI-SNR</div>
                    <div class="metric-value">${avgSiSnr.toFixed(2)}<span class="metric-unit">dB</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average STOI</div>
                    <div class="metric-value">${avgStoi.toFixed(3)}</div>
                </div>
            `;
        }
    }

    // Show separated sources
    const sourcesList = document.getElementById('sources-list');
    sourcesList.innerHTML = '';

    data.separated.forEach((sep, i) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';

        const metrics = data.metrics[`spk${i + 1}`] || {};
        const siSnr = metrics.si_snr ? `${metrics.si_snr.toFixed(2)} dB` : 'N/A';
        const stoi = metrics.stoi ? metrics.stoi.toFixed(3) : 'N/A';

        sourceItem.innerHTML = `
            <div class="source-header">
                <span class="source-title">Speaker ${i + 1}</span>
                <div class="source-metrics">
                    <span class="source-metric">SI-SNR: <span>${siSnr}</span></span>
                    <span class="source-metric">STOI: <span>${stoi}</span></span>
                </div>
            </div>
            <audio controls src="${sep.url}"></audio>
        `;

        sourcesList.appendChild(sourceItem);
    });

    // Show waveform comparison
    showWaveformComparison(data);

    // Show audio comparison
    const comparisonGrid = document.getElementById('comparison-grid');
    comparisonGrid.innerHTML = '';

    // Add mixture
    const mixItem = document.createElement('div');
    mixItem.className = 'comparison-item';
    mixItem.innerHTML = `
        <h4>Mixture (Input)</h4>
        <audio controls src="${state.selectedSample.audioUrl || state.selectedSample.mix}"></audio>
    `;
    comparisonGrid.appendChild(mixItem);

    // Add separated sources
    data.separated.forEach((sep, i) => {
        const item = document.createElement('div');
        item.className = 'comparison-item';
        item.innerHTML = `
            <h4>Separated - Speaker ${i + 1}</h4>
            <audio controls src="${sep.url}"></audio>
        `;
        comparisonGrid.appendChild(item);
    });

    // Sync playback
    document.getElementById('sync-playback').addEventListener('change', (e) => {
        const audios = comparisonGrid.querySelectorAll('audio');
        if (e.target.checked) {
            audios.forEach(audio => {
                audio.addEventListener('play', syncPlay);
                audio.addEventListener('pause', syncPause);
                audio.addEventListener('seeked', syncSeek);
            });
        } else {
            audios.forEach(audio => {
                audio.removeEventListener('play', syncPlay);
                audio.removeEventListener('pause', syncPause);
                audio.removeEventListener('seeked', syncSeek);
            });
        }
    });
}

// Show waveform comparison visualization
async function showWaveformComparison(data) {
    const section = document.getElementById('waveform-comparison-section');
    section.style.display = 'block';

    // Load all waveforms
    const waveforms = [];
    const labels = [];
    const colors = ['#0070f3', '#10b981', '#f59e0b', '#ef4444'];

    // Add mixture
    try {
        const mixturePath = state.selectedSample.mix;
        const response = await fetch(`${API_BASE}/waveform`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_path: mixturePath, points: 500 })
        });
        const waveformData = await response.json();
        if (waveformData.waveform) {
            waveforms.push(waveformData.waveform);
            labels.push({ name: 'Mixture', color: colors[0] });
        }
    } catch (error) {
        console.error('Error loading mixture waveform:', error);
    }

    // Add separated sources
    for (let i = 0; i < data.separated.length; i++) {
        try {
            const sep = data.separated[i];
            const response = await fetch(`${API_BASE}/waveform`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio_path: sep.path, points: 500 })
            });
            const waveformData = await response.json();
            if (waveformData.waveform) {
                waveforms.push(waveformData.waveform);
                labels.push({ name: `Speaker ${i + 1}`, color: colors[i + 1] });
            }
        } catch (error) {
            console.error(`Error loading separated waveform ${i}:`, error);
        }
    }

    // Draw stacked waveform
    drawStackedWaveform(waveforms, labels);

    // Draw individual waveforms
    drawIndividualWaveforms(waveforms, labels, data);
}

// Draw stacked waveform comparison
function drawStackedWaveform(waveforms, labels) {
    const canvas = document.getElementById('stacked-waveform-canvas');
    const container = document.getElementById('waveform-stack');
    const legend = document.getElementById('waveform-legend');
    
    if (!canvas || waveforms.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    // Draw each waveform with offset
    const rowHeight = height / waveforms.length;
    
    waveforms.forEach((waveform, index) => {
        const yOffset = index * rowHeight;
        const centerY = yOffset + rowHeight / 2;
        
        ctx.strokeStyle = labels[index].color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        const step = width / waveform.length;
        
        for (let i = 0; i < waveform.length; i++) {
            const x = i * step;
            const amplitude = waveform[i] * (rowHeight / 2 - 10);
            const y = centerY + amplitude;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Draw label
        ctx.fillStyle = labels[index].color;
        ctx.font = '12px Inter';
        ctx.fillText(labels[index].name, 10, yOffset + 20);
    });

    // Update legend
    legend.innerHTML = labels.map(l => `
        <div class="legend-item">
            <div class="legend-color" style="background-color: ${l.color}"></div>
            <span>${l.name}</span>
        </div>
    `).join('');
}

// Draw individual waveform rows
function drawIndividualWaveforms(waveforms, labels, data) {
    const container = document.getElementById('waveform-comparison-rows');
    container.innerHTML = '';

    waveforms.forEach((waveform, index) => {
        const row = document.createElement('div');
        row.className = 'waveform-row';

        const metrics = index === 0 ? null : (data.metrics[`spk${index}`] || {});
        const metricsHtml = metrics ? `
            <div class="waveform-row-metrics">
                SI-SNR: <span>${metrics.si_snr?.toFixed(2) || 'N/A'} dB</span> | 
                STOI: <span>${metrics.stoi?.toFixed(3) || 'N/A'}</span>
            </div>
        ` : '';

        row.innerHTML = `
            <div class="waveform-row-header">
                <span class="waveform-row-title" style="color: ${labels[index].color}">${labels[index].name}</span>
                ${metricsHtml}
            </div>
            <div class="waveform-chart">
                <canvas id="waveform-chart-${index}"></canvas>
            </div>
        `;

        container.appendChild(row);

        // Draw waveform on canvas
        setTimeout(() => {
            const canvas = document.getElementById(`waveform-chart-${index}`);
            if (!canvas) return;

            const chartContainer = canvas.parentElement;
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = chartContainer.getBoundingClientRect();
            
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const width = rect.width;
            const height = rect.height;
            const centerY = height / 2;

            ctx.clearRect(0, 0, width, height);

            // Draw center line
            ctx.strokeStyle = '#e5e5e5';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(width, centerY);
            ctx.stroke();

            // Draw waveform
            ctx.strokeStyle = labels[index].color;
            ctx.lineWidth = 2;
            ctx.beginPath();

            const step = width / waveform.length;
            
            for (let i = 0; i < waveform.length; i++) {
                const x = i * step;
                const amplitude = waveform[i] * (height / 2 - 10);
                const y = centerY + amplitude;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
        }, 0);
    });
}

function syncPlay(e) {
    const audios = document.querySelectorAll('#comparison-grid audio');
    audios.forEach(audio => {
        if (audio !== e.target) {
            audio.currentTime = e.target.currentTime;
            audio.play();
        }
    });
}

function syncPause(e) {
    const audios = document.querySelectorAll('#comparison-grid audio');
    audios.forEach(audio => {
        if (audio !== e.target) {
            audio.pause();
        }
    });
}

function syncSeek(e) {
    const audios = document.querySelectorAll('#comparison-grid audio');
    audios.forEach(audio => {
        if (audio !== e.target) {
            audio.currentTime = e.target.currentTime;
        }
    });
}

// Load samples table for dataset page
async function loadSamplesTable(datasetId, split) {
    try {
        const response = await fetch(`${API_BASE}/dataset/${datasetId}/samples?split=${split}&limit=50`);
        const data = await response.json();

        const tbody = document.querySelector('#samples-table tbody');
        tbody.innerHTML = '';

        data.samples.forEach(sample => {
            const row = document.createElement('tr');
            const numSpeakers = sample.s3 ? 3 : 2;
            
            row.innerHTML = `
                <td><code>${sample.id}</code></td>
                <td>5.0s</td>
                <td>${numSpeakers}</td>
                <td>
                    <button class="btn btn-secondary" onclick="playSample('${sample.mix}')">Play</button>
                </td>
            `;
            
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading samples table:', error);
    }
}

function playSample(audioPath) {
    const audio = new Audio(`/api/audio/dataset/${audioPath.replace(/^.*dataset\//, '')}`);
    audio.play();
}

// Loading overlay
function showLoading(show) {
    document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
}
