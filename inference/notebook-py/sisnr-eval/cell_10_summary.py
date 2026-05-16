import csv

summary = {}
for model_name in MODELS:
    csv_path = OUT_ROOT / f"{model_name}_per_file.csv"
    if not csv_path.exists():
        continue
    with csv_path.open() as f:
        r = csv.DictReader(f)
        vals = np.array([float(x["sisnr"]) for x in r])
    summary[model_name] = {
        "n":      int(len(vals)),
        "mean":   float(vals.mean()),
        "std":    float(vals.std()),
        "median": float(np.median(vals)),
        "p10":    float(np.percentile(vals, 10)),
        "p90":    float(np.percentile(vals, 90)),
        "min":    float(vals.min()),
        "max":    float(vals.max()),
    }

(OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))

print(f"{'model':<35} {'n':>5} {'mean':>8} {'std':>7} {'med':>7}")
print("-" * 68)
for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
    print(f"{name:<35} {s['n']:>5} {s['mean']:>8.3f} {s['std']:>7.3f} {s['median']:>7.3f}")
