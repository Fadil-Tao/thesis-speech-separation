# Dashboard

A modern, Vercel-inspired web dashboard for speech separation model evaluation and dataset monitoring.

## Features

### Model Evaluation
- Load and test trained models (SkiM and SkiM Attention)
- Select audio from dataset or upload your own
- Real-time inference with GPU acceleration
- Visualize waveforms
- Calculate SI-SNR and STOI metrics
- Side-by-side audio comparison with sync playback

### Dataset Monitoring
- View dataset statistics (total samples, duration, splits)
- Browse samples with audio preview
- Table view with filtering

### Supported Models
- SkiM 2-Speaker
- SkiM Attention 2-Speaker
- SkiM 3-Speaker
- SkiM Attention 3-Speaker

## Installation

1. Install dependencies:
```bash
pip install flask flask-cors
```

2. Make sure you have the trained models in the checkpoints directory:
```
checkpoints/
├── 2speaker/
│   ├── skim/best_model.pth
│   └── skim-attention/best_model.pth
└── 3speaker/
    ├── skim/best_model.pth
    └── skim-attention/best_model.pth
```

## Running the Dashboard

1. Start the backend server:
```bash
cd dashboard/backend
python app.py
```

The server will start on `http://localhost:5000`

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Model Evaluation

1. **Select a Model**: Choose from the dropdown (only models with checkpoints will be available)

2. **Select Audio**:
   - **From Dataset**: Choose a dataset and split, then click on a sample
   - **Upload**: Drag and drop or click to upload an audio file

3. **Run Evaluation**: Click the "Run Evaluation" button

4. **View Results**:
   - Metrics cards show SI-SNR and STOI scores
   - Separated sources with individual audio players
   - Comparison section with sync playback option

### Dataset Browser

1. Navigate to the "Dataset" page
2. Select a dataset and split
3. View samples in the table
4. Click "Play" to preview audio

## Design

The dashboard features a clean, modern design inspired by Vercel:
- Dark theme with high contrast
- Card-based layout
- Smooth transitions and hover effects
- Monospace fonts for technical data
- Responsive grid system

## API Endpoints

- `GET /api/models` - List available models
- `GET /api/datasets` - List available datasets
- `GET /api/dataset/{id}/samples` - Get dataset samples
- `POST /api/evaluate` - Run model evaluation
- `POST /api/waveform` - Get waveform data for visualization

## Troubleshooting

### Models not loading
- Check that checkpoint files exist in the correct directories
- Verify the model files are valid PyTorch checkpoints

### CUDA out of memory
- Reduce batch size in the backend code
- Use CPU inference (automatic fallback)

### Audio not playing
- Check browser console for errors
- Verify audio files are valid WAV format
- Ensure CORS is properly configured

## Development

To modify the dashboard:

1. **Frontend**: Edit files in `dashboard/frontend/`
   - `index.html` - Main HTML structure
   - `css/style.css` - Styling
   - `js/app.js` - Application logic

2. **Backend**: Edit `dashboard/backend/app.py`
   - Add new API endpoints
   - Modify model loading logic
   - Add new metrics calculations

3. **Restart the server** to see changes
