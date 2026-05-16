def normalize_length(x: np.ndarray) -> np.ndarray:
    if len(x) > TARGET_LEN:
        return x[:TARGET_LEN]
    if len(x) < TARGET_LEN:
        return np.pad(x, (0, TARGET_LEN - len(x)))
    return x


def load_wav(p: Path) -> np.ndarray:
    a, sr = sf.read(p)
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} from {p}"
    return normalize_length(a.astype(np.float32))


def si_snr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + EPS) * ref
    e_noise = est - s_target
    return float(
        10 * np.log10((np.dot(s_target, s_target) + EPS) / (np.dot(e_noise, e_noise) + EPS))
    )


def best_pit_sisnr(ests, refs):
    """Mean SI-SNR over speakers under best permutation."""
    best = -1e9
    for perm in permutations(range(len(refs))):
        v = float(np.mean([si_snr(ests[perm[i]], refs[i]) for i in range(len(refs))]))
        if v > best:
            best = v
    return best


@torch.no_grad()
def separate(enc, sep, dec, mix_np: np.ndarray):
    mix = torch.from_numpy(mix_np).unsqueeze(0).to(device)
    lengths = torch.tensor([mix.size(1)], dtype=torch.long, device=device)
    feats, flens = enc(mix, lengths)
    masked, _, _ = sep(feats, flens)
    ests = []
    for m in masked:
        wav, _ = dec(m, lengths)
        ests.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
    return [normalize_length(e) for e in ests]
