from autolightning import cc

from metalarena.configs.speech_commands.sc12_tiny_tcn import config
from metalarena.configs.chameleon_tcn_quant import create_quant_config


CHECKPOINT_PATH = "/space/ddenblanken/Projects/meta-learning-arena-new/src/metalarena/configs/speech_commands/test_wandb_lightning/430asogo/checkpoints/epoch=289-step=83520.ckpt"

# Hyperparameters below are taken from the best quantized run (wandb, start time 1742462053)

config["model"]["init_args"]["optimizer"]["dict_kwargs"].update(dict(
    lr=0.00002,
    weight_decay=0.00001028426368176649
))

config["model"]["init_args"]["criterion"]["init_args"]["label_smoothing"] = 0.06679972866852682

tcn_args = config["model"]["init_args"]["net"]["init_args"]["module"]["init_args"]

tcn_args["dropout"] = 0.013801863407105696
# During collect_stats_steps, cropping helps to improve initial training/validation
# accuracy and loss but afterwards the performance degrades faster. Peak validation
# performance is very similar.
tcn_args["input_length"] = 63
tcn_args["crop_hidden_states"] = True

# We need to decrease the learning rate right after collect_stats_steps stops, since
# the original learning rate is okay during stats collection but not afterwards.
config["model"]["init_args"]["lr_scheduler"]= cc(
    "autolightning.sched",
    lr_scheduler_path="cosine_annealing_warmup.CosineAnnealingWarmupRestarts",
    first_cycle_steps=45,
    cycle_mult=1.0,
    max_lr=0.00006930352894026597,
    min_lr=0.0000015031398266483083,
    warmup_steps=8,
    gamma=1.0
)


config["data"]["init_args"]["transforms"] = None
config["data"]["init_args"]["noise_p"] = 0.046980918087158735
config["data"]["init_args"]["max_noise_level"] = 0.3173847672938177

config["model"]["class_path"] = "autolightning.lm.brevitas.BrevitasClassifier"
config["model"]["init_args"].update(create_quant_config(collect_stats_steps=2000, limit_calibration_batches=None))
# The input quantizer of the best run collected statistics for fewer steps
config["model"]["init_args"]["in_quant"]["init_args"]["init_args"]["collect_stats_steps"] = 1500
config["model"]["init_args"]["net"] = cc(
    "autolightning.load",
    file_path=CHECKPOINT_PATH,
    state_dict_submodule="net._orig_mod",
    module=config["model"]["init_args"]["net"]["init_args"]["module"]
)

config["trainer"]["max_epochs"] = 45

config["seed_everything"] = 42

# Gradient clipping helps with quantized training
