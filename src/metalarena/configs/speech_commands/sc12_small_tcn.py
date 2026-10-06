from autolightning import cc

from metalarena.configs.speech_commands.sc12_data import create_data_config


def config(n_mffc: int = 32):
    config = {"data": create_data_config(n_mfcc=n_mffc)}

    config["model"] = cc(
        "autolightning.lm.Classifier",
        net=cc(
            "autolightning.compile",
            compiler_path="torch.compile",
            module=cc(
                "tcn_lib.TCN",
                input_size=n_mffc,
                channel_sizes=[20, 24, 24, 28],
                output_size=12,
                kernel_size=[5, 3, 3, 3],
                dropout=0.05,
                batch_norm=True,
                residual=True,
                weight_norm=False,
                bottleneck=False,
                zero_init_residual=True, # This really helps with achieving a good accuracy
                force_downsample=False
            )
        ),
        criterion=cc(
            "torch.nn.CrossEntropyLoss",
            label_smoothing=0.1
        ),
        optimizer=cc(
            "autolightning.optim",
            optimizer_path="AdamW",
            lr=0.015,
            weight_decay=2e-4
        ),
        lr_scheduler=cc(
            "autolightning.sched",
            lr_scheduler_path="MultiStepLR",
            milestones=[100, 130, 170],
            gamma=0.1
        )
    )

    config["trainer"] = dict(max_epochs=300)

    return config
