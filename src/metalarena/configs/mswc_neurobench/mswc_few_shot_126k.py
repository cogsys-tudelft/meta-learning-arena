from autolightning import cc

from metalarena.configs.mswc_neurobench.mswc_data import create_data_config


N_MFCC = 32


config = {"data": create_data_config(n_mfcc=N_MFCC, pre_load=True)}

config["data"]["class_path"] += "FewShot"
config["data"]["init_args"].update(
    dict(
        ways = 20,
        shots = 1,
        train_ways = 60,
        query_shots = 5,
        query_ways = 20,
        samples_per_class = {
            "train": 500,
            "val": 100,
            "test": 100
        },
        batch_transform_flattened = {
            "after": True
        }
    )
)
config["data"]["init_args"]["dataloaders"]["train"]["shuffle"] = False

config["model"] = cc(
    "autolightning.lm.Prototypical",
    average_support_embeddings=True,
    metric="euclidean-squared",
    net=cc(
        "autolightning.compile",
        compiler_path="torch.compile",
        module=cc(
            "tcn_lib.TCN",
            input_size=N_MFCC,
            channel_sizes=[48, 48, 64, 64, 64],
            kernel_size=3,
            output_size=-1,
            dropout=0.1,
            batch_norm=True,
            residual=True,
            weight_norm=False,
            zero_init_residual=True
        )
    ),
    optimizer=cc(
        "autolightning.optim",
        optimizer_path="Adam",
        lr=0.01,
        weight_decay=1e-4
    ),
    lr_scheduler=cc(
        "autolightning.sched",
        lr_scheduler_path="MultiStepLR",
        milestones=[35, 40, 45, 50],
        gamma=0.1
    )
)

config["trainer"] = dict(max_epochs=200)


if __name__ == "__main__":
    import yaml

    print(yaml.dump(config))
