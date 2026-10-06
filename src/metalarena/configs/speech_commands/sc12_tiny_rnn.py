from metalarena.configs.speech_commands.sc12_tiny_tcn import config

config["model"]["init_args"]["net"]["init_args"]["module"] = {
    "class_path": "torch_mate.models.ClassificationRNN",
    "init_args": {
        "input_size": 28,
        "hidden_size": 111,
        "output_size": 12,
    }
}
