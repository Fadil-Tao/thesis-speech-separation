results = {}  # model_name -> {"ests_aligned", "si_snr", "mean", "perm"}
for model_name, variant in MODELS.items():
    ckpt_path = CHECKPOINT_ROOT / model_name / 'best_model.pth'
    if not ckpt_path.exists():
        print(f"[skip] {model_name}: checkpoint missing at {ckpt_path}")
        continue
    print(f"Running {model_name} ({variant})...")
    model = build_model(variant)
    load_checkpoint(model, ckpt_path)
    ests = separate(model, mix_np)
    perm, vals, mean = pit_align(ests, refs)
    ests_aligned = [ests[perm[i]] for i in range(NUM_SPK)]
    results[model_name] = {
        "ests_aligned": ests_aligned,
        "si_snr": vals,
        "mean": mean,
        "perm": perm,
    }
    print(f"  perm={perm}, SI-SNR: {[f'{v:.2f}' for v in vals]}, mean={mean:.2f} dB")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
