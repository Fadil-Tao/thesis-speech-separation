random.seed(RANDOM_SEED)
mix_files = sorted((TEST_ROOT / 'mix').glob('*.wav'))
assert len(mix_files) > 0, f"No WAVs found under {TEST_ROOT / 'mix'}"
chosen = random.choice(mix_files)
file_id = chosen.stem
print(f"Chosen sample: {file_id}  (of {len(mix_files)} test files)")

mix_np = load_wav(chosen)
refs = [load_wav(TEST_ROOT / f's{i}' / f'{file_id}.wav') for i in range(1, NUM_SPK + 1)]
