from autolightning import cc

from metalarena.configs.mswc_neurobench.mswc_data import create_data_config


N_MFCC = 32

# Hparams taken from best sweep run for net with slightly more params
# https://wandb.ai/douwe/test_wandb_lightning/sweeps/3qbx088u/runs/vwzkvn2g/overview?nw=nwuserdouwe

config = {"data": create_data_config(n_mfcc=N_MFCC, pre_load=True)}

config["model"] = cc(
    "autolightning.lm.Classifier",
    net=cc(
        "autolightning.compile",
        compiler_path="torch.compile",
        module=cc(
            "tcn_lib.TCN",
            input_size=N_MFCC,
            channel_sizes=[(48, 64), 64, 64, 64, 64],
            output_size=100,
            kernel_size=[3, 3, 3, 3, 3],
            dropout=0.09,
            batch_norm=True,
            residual=True,
            weight_norm=False,
            zero_init_residual=True,
            pre_pad_with_receptive_field=True
        )
    ),
    criterion=cc(
        "torch.nn.CrossEntropyLoss",
        label_smoothing=0.03113687838334804
    ),
    optimizer=cc(
        "autolightning.optim",
        optimizer_path="Adam",
        lr=0.0057194646354298455,
        weight_decay=0.000004620938823559334
    ),
    lr_scheduler=cc(
        "autolightning.sched",
        lr_scheduler_path="MultiStepLR",
        milestones=[35, 40, 45, 50],
        gamma=0.3921523594337123,
    )
)

config["trainer"] = dict(max_epochs=110, check_val_every_n_epoch=10)
