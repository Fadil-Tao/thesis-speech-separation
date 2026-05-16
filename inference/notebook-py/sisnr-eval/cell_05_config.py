DRIVE_ROOT = Path('/content/drive/MyDrive/Tugas Akhir')
CHECKPOINT_ROOT = DRIVE_ROOT / 'Checkpoints' / '100 epochs' / '3speaker'
TEST_ROOT = DRIVE_ROOT / 'Dataset' / 'synthetic-titml-idn-mix' / 'raw' / 'TITML-3spk' / 'test'
OUT_ROOT = DRIVE_ROOT / 'Inference' / 'sisnr-eval'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
TARGET_DURATION = 5.0
TARGET_LEN = int(SAMPLE_RATE * TARGET_DURATION)
NUM_SPK = 3
EPS = 1e-8

# Cap for smoke testing — set to None for full 3600.
MAX_FILES = None
# Skip a model if its CSV already complete.
SKIP_DONE = True

MODELS = {
    "skim":                            "skim",
    "skim-transfer":                   "skim",
    "skim-attention-v3":               "v3",
    "skim-attention-v3-reg-transfer":  "v3",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:          {device}")
print(f"checkpoint root: {CHECKPOINT_ROOT}")
print(f"test root:       {TEST_ROOT}")
print(f"out root:        {OUT_ROOT}")
