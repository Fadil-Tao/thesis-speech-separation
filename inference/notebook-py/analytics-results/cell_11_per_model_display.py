t = np.arange(TARGET_LEN) / SAMPLE_RATE

for model_name, res in results.items():
    display(Markdown(f"---\n## Model: `{model_name}`"))
    display(Markdown(f"Permutation (est-idx aligned to ref s1,s2,s3): `{res['perm']}`"))

    df = pd.DataFrame(
        {"SI-SNR (dB)": [*res["si_snr"], res["mean"]]},
        index=["s1", "s2", "s3", "mean"],
    )
    display(df.style.format("{:.2f}"))

    display(Markdown("### Estimated sources"))
    for i, est in enumerate(res["ests_aligned"], start=1):
        display(Markdown(f"**est s{i}** (aligned to ref s{i})"))
        display(Audio(est, rate=SAMPLE_RATE))

    fig, axes = plt.subplots(4, 2, figsize=(14, 9), sharex=True)
    axes[0, 0].plot(t, mix_np, color='gray', lw=0.6)
    axes[0, 0].set_title("Mixture")
    axes[0, 1].axis('off')
    for i in range(NUM_SPK):
        axes[i + 1, 0].plot(t, refs[i], color='C0', lw=0.6)
        axes[i + 1, 0].set_title(f"Ref s{i + 1}")
        axes[i + 1, 1].plot(t, res["ests_aligned"][i], color='C3', lw=0.6)
        axes[i + 1, 1].set_title(f"Est s{i + 1} (SI-SNR = {res['si_snr'][i]:.2f} dB)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"{model_name} — {file_id}")
    fig.tight_layout()
    plt.show()
