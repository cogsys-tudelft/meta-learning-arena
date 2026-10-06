from metalarena.configs.speech_commands.sc12_small_tcn import config

config["model"]["init_args"]["net"]["init_args"]["module"]["init_args"]["output_size"] = 35
config["data"]["class_path"] = "metalarena.autodatasets.SpeechCommandsV2_35C"

del config["data"]["init_args"]["noise_p"]
del config["data"]["init_args"]["max_noise_level"]
