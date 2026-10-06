from autolightning import cc

from metalarena.configs.speech_commands.sc12_data import create_data_config


config = {"data": create_data_config(task_representation='raw', take_log=False)}

config["model"] = cc(
    "autolightning.lm.Classifier",
    net=cc(
        "autolightning.compile",
        compiler_path="torch.compile",
        module=cc(
            "tcn_lib.TCN",
            input_size=1,
            channel_sizes=[32]*6 + [48]*6,
            output_size=12,
            kernel_size=3,
            dropout=0.025,
            batch_norm=True,
            residual=True,
            weight_norm=False,
            bottleneck=False,
            input_length=16000,
            crop_hidden_states=True,
            zero_init_residual=True, # This really helps with achieving a good accuracy
            force_downsample=False
        )
    )
)

config["trainer"] = dict(max_epochs=300)

config["optimizer"] = cc(
    "torch.optim.AdamW",
    init_args=dict(
        lr=0.015,
        weight_decay=2e-4
    )
)

config["lr_scheduler"] = cc(
    "torch.optim.lr_scheduler.MultiStepLR",
    init_args=dict(
        milestones=[100, 130, 170],
        gamma=0.1
    )
)
