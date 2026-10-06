from autolightning import cc

from metalarena.configs.speech_commands.sc12_data import create_data_config


# Set train batch size to 1 to make sure the model can be trained within the memory of a single GPU
config = {"data": create_data_config(task_representation='raw', take_log=False, train_batch_size=1)}

config["model"] = cc(
    "autolightning.lm.Classifier",
    net=cc(
        "autolightning.compile",
        compiler_path="torch.compile",
        module=cc(
            "torch_mate.models.ClassificationRNN",
            input_size=1,
            hidden_size=338, # RNN is 338 # LSTM is 170
            output_size=12
        )
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
