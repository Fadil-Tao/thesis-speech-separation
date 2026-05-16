def build_model(variant: str) -> ESPnetEnhancementModel:
    enc_cfg = MODEL_CONFIG_BASE["encoder"]
    dec_cfg = MODEL_CONFIG_BASE["decoder"]
    sep_cfg = dict(MODEL_CONFIG_BASE["separator"])

    encoder = ConvEncoder(**enc_cfg)
    decoder = ConvDecoder(**dec_cfg)

    if variant == "skim":
        separator = SkiMSeparator(**sep_cfg)
    elif variant == "skim-attention":
        separator = SkiMAttentionV3Separator(num_heads=ATTENTION_NUM_HEADS, **sep_cfg)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=[PITSolver(criterion=SISNRLoss())],
        loss_type="si_snr",
    )
    return model.to(device)


def load_checkpoint(model: ESPnetEnhancementModel, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
