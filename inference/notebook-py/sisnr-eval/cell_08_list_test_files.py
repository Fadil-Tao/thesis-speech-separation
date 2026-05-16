mix_files = sorted((TEST_ROOT / 'mix').glob('*.wav'))
if MAX_FILES is not None:
    mix_files = mix_files[:MAX_FILES]
print(f"Test mixes: {len(mix_files)}")
assert len(mix_files) > 0, f"No WAVs found under {TEST_ROOT / 'mix'}"
