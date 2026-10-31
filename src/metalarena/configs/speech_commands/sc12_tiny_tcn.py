"""This tiny configuration refers to a network that can operate
within the subsection mode of the Chameleon chip, meaning that
it has less than or equal to 16384 weights and 512 biases."""

from metalarena.configs.speech_commands.sc12_small_tcn import config

config = config(n_mffc=28)

# Hyperparameters below are taken from the best FP32 run (wandb, start time 1742457362)

net_init_args = config["model"]["init_args"]["net"]["init_args"]["module"]["init_args"]

net_init_args["channel_sizes"] = [24, 24, 24, 24]
net_init_args["kernel_size"] = [4, 3, 3, 3]
net_init_args["dropout"] = 0.010436289712459978

config["model"]["init_args"]["criterion"]["init_args"]["label_smoothing"] = 0.10362673455779092

config["model"]["init_args"]["optimizer"]["dict_kwargs"].update(dict(
    lr=0.0059256432000385755,
    weight_decay=0.0002644742578702141
))

data_init_args = config["data"]["init_args"]

data_init_args["noise_p"] = 0.04638662845578956
data_init_args["max_noise_level"] = 0.8405225525059595
data_init_args["dataloaders"]["defaults"]["batch_size"] = 128

config["seed_everything"] = 42
