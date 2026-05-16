for model_name, variant in MODELS.items():
    ckpt_path = CHECKPOINT_ROOT / model_name / 'best_model.pth'
    if not ckpt_path.exists():
        print(f"[skip] {model_name}: checkpoint missing at {ckpt_path}")
        continue

    print(f"\n=== {model_name} ({variant}) ===")
    print(f"checkpoint: {ckpt_path}")

    out_dirs = {i: OUT_ROOT / model_name / f's{i}' for i in range(1, NUM_SPK + 1)}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    model = build_model(variant)
    load_checkpoint(model, ckpt_path)

    written, skipped = 0, 0
    for mix_path in tqdm(mix_files, desc=model_name):
        file_id = mix_path.stem
        targets = [out_dirs[i] / f'{file_id}.wav' for i in range(1, NUM_SPK + 1)]
        if all(t.exists() for t in targets):
            skipped += 1
            continue

        mix_np = load_mix(mix_path)
        ests = separate(model, mix_np)
        for i, wav in enumerate(ests, start=1):
            sf.write(out_dirs[i] / f'{file_id}.wav', wav, SAMPLE_RATE)
        written += 1

    print(f"  written: {written}, skipped (already done): {skipped}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
