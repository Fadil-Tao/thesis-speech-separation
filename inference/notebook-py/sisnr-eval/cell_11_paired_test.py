# Paired comparison: per-file SI-SNR diffs + paired t-test.
# Tests whether mean diff between two models is significantly non-zero.
import csv
from scipy import stats

def load_csv(name):
    with (OUT_ROOT / f"{name}_per_file.csv").open() as f:
        r = csv.DictReader(f)
        return {x["file_id"]: float(x["sisnr"]) for x in r}

PAIRS = [
    ("skim",          "skim-attention-v3"),                # base vs base
    ("skim-transfer", "skim-attention-v3-reg-transfer"),   # transfer vs transfer
    ("skim",          "skim-transfer"),                    # base vs transfer (skim)
    ("skim-attention-v3", "skim-attention-v3-reg-transfer"),
]

for a_name, b_name in PAIRS:
    a, b = load_csv(a_name), load_csv(b_name)
    keys = sorted(set(a) & set(b))
    if not keys:
        print(f"no overlap: {a_name} vs {b_name}")
        continue
    diffs = np.array([b[k] - a[k] for k in keys])
    t, p = stats.ttest_rel([b[k] for k in keys], [a[k] for k in keys])
    print(f"{b_name} − {a_name}  (n={len(keys)})")
    print(f"  mean diff: {diffs.mean():+.4f} dB  std {diffs.std():.3f}")
    print(f"  paired t  : t={t:.3f}  p={p:.3g}  {'SIGNIF' if p<0.05 else 'n.s.'}")
    win = (diffs > 0).sum()
    print(f"  {b_name} wins on {win}/{len(keys)} files ({100*win/len(keys):.1f}%)\n")
