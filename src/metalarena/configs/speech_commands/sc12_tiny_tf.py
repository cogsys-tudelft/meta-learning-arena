from metalarena.configs.speech_commands.sc12_tiny_tcn import config

config["model"]["init_args"]["net"]["init_args"]["module"] = {
    "class_path": "metalarena.models.TransformerClassifier",
    "init_args": {
        "input_dim": 28,
        "nhead": 2,
        "d_model": 20,
        "num_classes": 12,
        "dim_feedforward": 80
    }
}
