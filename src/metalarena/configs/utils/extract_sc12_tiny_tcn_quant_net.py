from autolightning import auto_main

from brevitas_utils.export import get_quant_state_dict, save_quant_state_dict

from metalarena.configs.speech_commands.sc12_tiny_tcn_quant import config


def export_quant_state_dict(ckpt_path: str, save_path: str, as_numpy: bool=True):
    cli = auto_main(config, run=False)

    _, model, _ = cli

    import torch
    # load state dict into model.net

    with open(ckpt_path, 'rb') as f:
        checkpoint = torch.load(f)

    # replace all "*.relu.*" with "*.act_fn.*" in the checkpoint state dict, since
    # newer versions of Brevitas have changed the naming convention for activation functions
    new_state_dict = {}

    for k, v in checkpoint['state_dict'].items():
        if '.relu.' in k:
            new_k = k.replace('.relu.', '.act_fn.')
        else:
            new_k = k
            
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)

    state_dict = get_quant_state_dict(
        model.net,
        input_shape_or_quant_tensor=(1, 28, 63),
        as_numpy=as_numpy
    )

    save_quant_state_dict(state_dict, f"{save_path}.qsd.pkl")
