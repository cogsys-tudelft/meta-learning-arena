"""This tiny configuration refers to a network that can operate
within the subsection mode of the Chameleon chip, meaning that 
it has less than or equal to 16384 weights and 512 biases."""

from metalarena.configs.speech_commands.sc12_small_tcn import config

config = config(n_mffc=28)

net_init_args = config["model"]["init_args"]["net"]["init_args"]["module"]["init_args"]

net_init_args["channel_sizes"] = [24, 24, 24, 24]
net_init_args["kernel_size"] = [4, 3, 3, 3]
net_init_args["dropout"] = 0.0

config["model"]["init_args"]["criterion"]["init_args"]["label_smoothing"] = 0.0
