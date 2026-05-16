def normalize_length(x: np.ndarray) -> np.ndarray:
    if len(x) > TARGET_LEN:
        return x[:TARGET_LEN]
    if len(x) < TARGET_LEN:
        return np.pad(x, (0, TARGET_LEN - len(x)))
    return x


def load_mix(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} from {path}"
    return normalize_length(audio.astype(np.float32))


@torch.no_grad()
def separate(model: ESPnetEnhancementModel, mix_np: np.ndarray) -> list:
    """Return list of NUM_SPK numpy waveforms, shape (TARGET_LEN,)."""
    mix = torch.from_numpy(mix_np).unsqueeze(0).to(device)  # (1, T)
    lengths = torch.tensor([mix.size(1)], dtype=torch.long, device=device)
    feats, flens = model.encoder(mix, lengths)
    masked, _, _ = model.separator(feats, flens)  # list[Tensor]
    ests = []
    for m in masked:
        wav, _ = model.decoder(m, lengths)  # decoder needs sample-domain lengths
        ests.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
    return ests
