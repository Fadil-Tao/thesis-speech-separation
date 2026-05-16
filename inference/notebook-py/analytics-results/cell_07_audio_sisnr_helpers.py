def normalize_length(x: np.ndarray) -> np.ndarray:
    if len(x) > TARGET_LEN:
        return x[:TARGET_LEN]
    if len(x) < TARGET_LEN:
        return np.pad(x, (0, TARGET_LEN - len(x)))
    return x


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} from {path}"
    return normalize_length(audio.astype(np.float32))


@torch.no_grad()
def separate(model: ESPnetEnhancementModel, mix_np: np.ndarray) -> list:
    mix = torch.from_numpy(mix_np).unsqueeze(0).to(device)
    lengths = torch.tensor([mix.size(1)], dtype=torch.long, device=device)
    feats, flens = model.encoder(mix, lengths)
    masked, _, _ = model.separator(feats, flens)
    ests = []
    for m in masked:
        wav, _ = model.decoder(m, lengths)  # decoder needs sample-domain lengths
        ests.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
    return ests


def si_snr_db(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    dot = np.dot(est, ref)
    ref_energy = np.dot(ref, ref) + eps
    s_target = dot / ref_energy * ref
    e_noise = est - s_target
    return 10.0 * np.log10((np.dot(s_target, s_target) + eps) / (np.dot(e_noise, e_noise) + eps))


def pit_align(ests: list, refs: list) -> tuple:
    """Brute-force PIT over 3! perms. Returns (best_perm, per_ref_si_snr_db, mean_si_snr)."""
    best = None
    for perm in permutations(range(len(ests))):
        vals = [si_snr_db(ests[perm[i]], refs[i]) for i in range(len(refs))]
        mean = float(np.mean(vals))
        if best is None or mean > best[2]:
            best = (perm, vals, mean)
    return best
