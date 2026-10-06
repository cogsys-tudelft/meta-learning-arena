from autolightning import cc

from metalarena.configs.speech_commands.sc12_raw_audio_lstm import config

config["model"]["init_args"]["net"]["init_args"]["module"] = cc(
    "torch_mate.models.ClassificationGRU",
    input_size=1,
    hidden_size=195,
    output_size=12
)
