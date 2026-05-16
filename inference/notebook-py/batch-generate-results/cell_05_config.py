DRIVE_ROOT = Path('/content/drive/MyDrive/Tugas Akhir')
CHECKPOINT_ROOT = DRIVE_ROOT / 'Checkpoints' / '100 epochs' / '3speaker'
TEST_ROOT = DRIVE_ROOT / 'Dataset' / 'synthetic-titml-idn-mix' / 'raw' / 'TITML-3spk' / 'test'
OUT_ROOT = DRIVE_ROOT / 'Inference'

SAMPLE_RATE = 16000
TARGET_DURATION = 5.0
TARGET_LEN = int(SAMPLE_RATE * TARGET_DURATION)
NUM_SPK = 3

# Cap for smoke testing — set to None for full ~3600.
MAX_FILES = None

MODELS = {
    "skim":                    "skim",
    "skim-transfer":           "skim",
    "skim-attention":          "skim-attention",
    "skim-attention-transfer": "skim-attention",
}

MODEL_CONFIG_BASE = {
    "encoder": {"channel": 256, "kernel_size": 32, "stride": 16},
    "decoder": {"channel": 256, "kernel_size": 32, "stride": 16},
    "separator": {
        "input_dim": 256,
        "causal": False,
        "num_spk": NUM_SPK,
        "predict_noise": False,
        "nonlinear": "relu",
        "layer": 4,
        "unit": 256,
        "segment_size": 20,
        "dropout": 0.2,
        "mem_type": "hc",
        "seg_overlap": False,
    },
}
ATTENTION_NUM_HEADS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Checkpoint root: {CHECKPOINT_ROOT}")
print(f"Test root:       {TEST_ROOT}")
print(f"Output root:     {OUT_ROOT}")
