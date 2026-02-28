#!/bin/bash
# Install all required dependencies for training

echo "Installing dependencies for speech separation training..."

# Activate virtual environment
source /home/dl-1/hadad/speech-separation/.venv/bin/activate

# Install PyTorch with CUDA
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install ESPnet and dependencies
pip install espnet==202304
pip install espnet-model-zoo

# Install audio processing
pip install soundfile librosa==0.9.2

# Install evaluation metrics
pip install mir_eval pesq pystoi

# Install visualization
pip install matplotlib tqdm

echo "Installation complete!"
