#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import re
from pathlib import Path
from typing import Tuple, Any

import torch
import numpy as np
from autolightning import auto_main
from brevitas_utils.export import get_quant_state_dict, save_quant_state_dict


CONFIG_PATH: str = "quant_zero_shot_103k_28k.yaml"


def parse_checkpoint_info(ckpt_path: Path) -> Tuple[str, str]:
    """
    Extract run id and step from a Lightning checkpoint path.

    Example:
    .../h4hhna3b/checkpoints/epoch=0-step=107500.ckpt
    -> ("h4hhna3b", "107500")
    """
    run_id: str = ckpt_path.parents[1].name

    match = re.search(r"step=(\d+)", ckpt_path.name)
    if match is None:
        raise ValueError(f"Could not extract step from checkpoint name: {ckpt_path.name}")

    step: str = match.group(1)
    return run_id, step


def main() -> None:
    parser = argparse.ArgumentParser(description="Export quantized state dicts from a Lightning checkpoint")
    parser.add_argument("--ckpt_path", type=Path, help="Path to .ckpt file")
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path.resolve()
    run_id, step = parse_checkpoint_info(ckpt_path)

    # Hide CLI args for this file from Lightning to avoid a massive printout
    sys.argv = [sys.argv[0]]

    _, model, _ = auto_main(CONFIG_PATH, run=False)

    with ckpt_path.open("rb") as f:
        checkpoint: dict[str, Any] = torch.load(f, map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"])

    query_sample_embedder_state_dict = get_quant_state_dict(
        model.query_sample_embedder[2],
        input_shape_or_quant_tensor=(1, 31, 827),
        as_numpy=True,
    )

    embedding_weight = model.sample_embedder[0].weight.cpu().detach().numpy()

    sample_embedder_state_dict = get_quant_state_dict(
        model.sample_embedder[2],
        input_shape_or_quant_tensor=(1, 160),
        as_numpy=True,
    )

    # The net module does not have an input quantizer, so we take the output from the
    # sample embedder as input to the net, as this output has the correct scale.
    out = model.sample_embedder[2].cpu()(torch.randn(1, 160))
    inp_for_net = torch.cat([out]*6, dim=1)

    net_state_dict = get_quant_state_dict(
        model.net,
        input_shape_or_quant_tensor=inp_for_net,
        as_numpy=True,
    )

    postfix: str = f"{run_id}_step={step}.qsd.pkl"

    save_quant_state_dict(net_state_dict, f"zsl_fsc_net_{postfix}")
    save_quant_state_dict(
        query_sample_embedder_state_dict,
        f"zsl_fsc_query_sample_embedder_{postfix}",
    )
    save_quant_state_dict(
        sample_embedder_state_dict,
        f"zsl_fsc_sample_embedder_{postfix}",
    )
    np.save(f"zsl_fsc_embedding_weight_{postfix}.npy", embedding_weight)

    print("\nSuccesfully saved quant state dicts!")


if __name__ == "__main__":
    main()
