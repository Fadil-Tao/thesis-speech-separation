display(Markdown(f"## Sample: `{file_id}`"))
display(Markdown("### Mixture"))
display(Audio(mix_np, rate=SAMPLE_RATE))
display(Markdown("### Reference sources (ground truth)"))
for i, ref in enumerate(refs, start=1):
    display(Markdown(f"**s{i}**"))
    display(Audio(ref, rate=SAMPLE_RATE))
