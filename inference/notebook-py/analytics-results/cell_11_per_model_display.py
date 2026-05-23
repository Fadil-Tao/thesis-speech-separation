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

    speaker_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # s1=blue, s2=orange, s3=green

    fig, axes = plt.subplots(4, 2, figsize=(14, 9), sharex=False)

    # Mixture: overlay each ref source colored so composition visible
    axes[0, 0].plot(t, mix_np, color='black', lw=0.4, alpha=0.4, label='Mixture')
    for i in range(NUM_SPK):
        axes[0, 0].plot(t, refs[i], color=speaker_colors[i], lw=0.5, alpha=0.7, label=f's{i+1}')
    axes[0, 0].set_title("Mixture (overlay per speaker)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend(loc='upper right', fontsize=8, ncol=4)
    axes[0, 1].axis('off')

    for i in range(NUM_SPK):
        est_i = res["ests_aligned"][i]
        ref_rms = np.sqrt(np.mean(refs[i] ** 2)) + 1e-8
        est_rms = np.sqrt(np.mean(est_i ** 2)) + 1e-8
        est_plot = est_i * (ref_rms / est_rms)
        c = speaker_colors[i]
        axes[i + 1, 0].plot(t, refs[i], color=c, lw=0.6)
        axes[i + 1, 0].set_title(f"Ref s{i + 1}")
        axes[i + 1, 0].set_ylabel("Amplitude")
        axes[i + 1, 1].plot(t, est_plot, color=c, lw=0.6, linestyle='--')
        axes[i + 1, 1].set_title(f"Est s{i + 1} (SI-SNR = {res['si_snr'][i]:.2f} dB)")
        axes[i + 1, 1].set_ylabel("Amplitude")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Time (s)")
    axes[0, 1].set_xlabel("")  # hidden subplot, no label needed
    fig.suptitle(f"{model_name} — {file_id}")
    fig.tight_layout()
    plt.show()
