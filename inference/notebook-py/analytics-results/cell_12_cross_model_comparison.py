if results:
    rows = []
    for name, res in results.items():
        rows.append({
            "model": name,
            "si_snr_s1": res["si_snr"][0],
            "si_snr_s2": res["si_snr"][1],
            "si_snr_s3": res["si_snr"][2],
            "mean": res["mean"],
        })
    summary = pd.DataFrame(rows).set_index("model")
    display(Markdown("## Cross-model SI-SNR comparison"))
    display(summary.style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(8, 4))
    summary["mean"].plot(kind="bar", ax=ax, color="C2")
    ax.set_ylabel("Mean SI-SNR (dB)")
    ax.set_title(f"Mean SI-SNR per model — sample {file_id}")
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No model results to compare.")
