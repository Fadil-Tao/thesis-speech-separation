print("\n=== Summary ===")
for model_name in MODELS:
    base = OUT_ROOT / model_name
    counts = {f's{i}': len(list((base / f's{i}').glob('*.wav'))) for i in range(1, NUM_SPK + 1)}
    print(f"{model_name}: {counts}")
