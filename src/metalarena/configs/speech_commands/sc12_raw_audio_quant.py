from autolightning import cc

from metalarena.configs.speech_commands.sc12_raw_audio_tcn import config
from metalarena.configs.chameleon_tcn_quant import create_quant_config

# Gradient clipping really helps with quantized training; while the training accuracy
# is not affected, the validation accuracy just starts dropping significantly without
# it after some time.

CHECKPOINT_PATH = "/space/ddenblanken/Projects/meta-learning-arena-new/src/metalarena/configs/speech_commands/test_wandb_lightning/nciv4c90/checkpoints/epoch=249-step=36000.ckpt"

model_args = config["model"]["init_args"]

config["optimizer"]["init_args"]["lr"] = 0.001
config["optimizer"]["init_args"]["weight_decay"] = 0.0

tcn_args = model_args["net"]["init_args"]["module"]["init_args"]

tcn_args["dropout"] = 0.0

model_args["exclude_no_grad"] = True
config["lr_scheduler"]["init_args"]["milestones"] = [22, 40]
config["data"]["init_args"].update(dict(
    transforms=None,
    noise_p=0.0
))

config["model"]["class_path"] = "autolightning.lm.brevitas.BrevitasClassifier"
model_args.update(create_quant_config(ignore_overflow="zero_gradients_outside_range", collect_stats_steps=500))
model_args["net"] = cc(
    "autolightning.load",
    file_path=CHECKPOINT_PATH,
    state_dict_submodule="net._orig_mod",
    module=model_args["net"]["init_args"]["module"]
)

config["trainer"]["max_epochs"] = 45
