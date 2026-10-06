from autolightning import cc

from metalarena.configs.mswc_neurobench.mswc_supervised_126k import config
from metalarena.configs.chameleon_tcn_quant import create_quant_config


# Best run from the following sweep (https://wandb.ai/douwe/test_wandb_lightning/sweeps/3qbx088u?nw=nwuserdouwe)
# CHECKPOINT_PATH = "/space2/ddenblanken/Experiments/meta-learning-arena/test_wandb_lightning/vwzkvn2g/checkpoints/epoch=49-step=9750.ckpt"

CHECKPOINT_PATH = "/space2/ddenblanken/Experiments/meta-learning-arena/test_wandb_lightning/ya3fbgh5/checkpoints/epoch=69-step=13650.ckpt"

config["model"]["init_args"]["optimizer"]["dict_kwargs"].update(dict(
    lr=0.00006490673720899333/5,
    weight_decay=1.437550863037048e-7
))

tcn_args = config["model"]["init_args"]["net"]["init_args"]["module"]["init_args"]

tcn_args["dropout"] = 0.07526561929400599

config["model"]["init_args"]["criterion"]["init_args"]["label_smoothing"] = 0.039244647093823866

config["model"]["init_args"]["lr_scheduler"]["dict_kwargs"]["gamma"] = 0.12044020483625792
config["model"]["init_args"]["lr_scheduler"]["dict_kwargs"]["milestones"] = [10, 13, 15]
config["data"]["init_args"]["transforms"] = None

config["model"]["class_path"] = "autolightning.lm.brevitas.BrevitasClassifier"
config["model"]["init_args"].update(create_quant_config(
    skip_linear_layers=True,
    collect_stats_steps=3*190,
    limit_calibration_batches=None, subnet_quant_config=["net.1"]))
config["model"]["init_args"]["net"] = cc("autolightning.sequential", modules=[
        cc("torch.nn.ZeroPad1d", padding=[125, 0]),
        cc("autolightning.load",
    file_path=CHECKPOINT_PATH,
    state_dict_submodule="net._orig_mod",
    module=config["model"]["init_args"]["net"]["init_args"]["module"])
    ])

config["trainer"]["max_epochs"] = 25
config["trainer"]["check_val_every_n_epoch"] = 3
