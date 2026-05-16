for model_name, variant in MODELS.items():
    ckpt_path = CHECKPOINT_ROOT / model_name / 'best_model.pth'
    if not ckpt_path.exists():
        print(f"[skip] {model_name}: missing {ckpt_path}")
        continue

    csv_path = OUT_ROOT / f"{model_name}_per_file.csv"
    if SKIP_DONE and csv_path.exists():
        n_done = sum(1 for _ in csv_path.open()) - 1
        if n_done >= len(mix_files):
            print(f"[done] {model_name}: {n_done} rows, skip")
            continue

    print(f"\n=== {model_name} ({variant}) ===")
    enc, sep, dec = build_model(variant, ckpt_path)

    sisnrs = []
    csv_f = csv_path.open("w", buffering=1)
    csv_f.write("file_id,sisnr\n")
    for mp in tqdm(mix_files, desc=model_name):
        fid = mp.stem
        mix_np = load_wav(mp)
        refs = [load_wav(TEST_ROOT / f's{i}' / f'{fid}.wav') for i in range(1, NUM_SPK + 1)]
        ests = separate(enc, sep, dec, mix_np)
        s = best_pit_sisnr(ests, refs)
        sisnrs.append(s)
        csv_f.write(f"{fid},{s:.4f}\n")
    csv_f.close()

    arr = np.array(sisnrs)
    print(f"  mean SI-SNR : {arr.mean():.3f} ± {arr.std():.3f} dB  (median {np.median(arr):.3f})")

    del enc, sep, dec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
