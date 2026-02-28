#!/bin/bash
# Resume training from best checkpoint

echo "Resuming training from best checkpoint..."

# Activate virtual environment
source /home/dl-1/hadad/speech-separation/.venv/bin/activate

# Run training script
cd /home/dl-1/hadad/speech-separation
python train/2speaker/skim/train_skim_2spk.py --resume-from checkpoints/2speaker/skim/best_model.pth
