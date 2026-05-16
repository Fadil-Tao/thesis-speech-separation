def build_model(variant: str, ckpt_path: Path):
    """Build encoder/separator/decoder from per-ckpt config.json and load weights.

    Returns (encoder, separator, decoder) all on `device`, eval mode.
    """
    cfg = json.loads((ckpt_path.parent / "config.json").read_text())["model_config"]
    enc = ConvEncoder(**cfg["encoder"])
    dec = ConvDecoder(**cfg["decoder"])
    sep_cfg = dict(cfg["separator"])
    if variant == "skim":
        sep = SkiMSeparator(**sep_cfg)
    elif variant == "v3":
        sep = SkiMAttentionV3Separator(**sep_cfg)
    else:
        raise ValueError(variant)

    enc, sep, dec = enc.to(device).eval(), sep.to(device).eval(), dec.to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    enc_sd = {k.replace("encoder.", "", 1): v for k, v in sd.items() if k.startswith("encoder.")}
    sep_sd = {k.replace("separator.", "", 1): v for k, v in sd.items() if k.startswith("separator.")}
    dec_sd = {k.replace("decoder.", "", 1): v for k, v in sd.items() if k.startswith("decoder.")}
    enc.load_state_dict(enc_sd)
    sep.load_state_dict(sep_sd)
    dec.load_state_dict(dec_sd)
    return enc, sep, dec
