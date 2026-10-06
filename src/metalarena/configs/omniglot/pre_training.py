from dotmap import DotMap

permute = False
random_rotation = [0, 90, 180, 270]
num_pre_train_classes = 1028
augment_train = False

cfg = DotMap()

cfg.learner.name = "SupervisedLearner"
cfg.learner.cfg.classification = True

cfg.criterion.name = "CrossEntropyLoss"

cfg.model.name = "tcn_lib.TCN"
cfg.model.cfg = {
    "input_size": 1,
    "channel_sizes": [32]*3+[48]*4,
    "output_size": num_pre_train_classes * len(random_rotation),
    "kernel_size": 5,
    "dropout": 0.025,
    "batch_norm": True,
    "residual": True,
    "weight_norm": False,
    "zero_init_residual": False
}
cfg.model.extra.compile.name = "torch.compile"

cfg.optimizer.name = "SGD"
cfg.optimizer.cfg = DotMap({"lr": 0.18, "momentum": 0.9, "weight_decay": 0.0005})
cfg.lr_scheduler.scheduler.name = "MultiStepLR"
cfg.lr_scheduler.scheduler.cfg = DotMap(
    {"milestones": [100, 200, 300], "gamma": 0.1}
)

cfg.training = {
    "max_epochs": 200
}

cfg.dataset.name = "metalarena.autodatasets.OmniglotSupervised"
cfg.dataset.cfg.val_percentage = 0.1
cfg.dataset.cfg.class_val_percentage = (1200-num_pre_train_classes)/1200
cfg.dataset.cfg.random_rotation = random_rotation
cfg.dataset.cfg.shuffle_classes = False
cfg.dataset.cfg.variant = "Vinyals-split"
cfg.dataset.extra.pre_load = ['train', 'val']

cfg.dataloaders = {
    "default": {
        "batch_size": 1024,
        "num_workers": 16,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "pin_memory": True,
    },
    "train": {
        "shuffle": True
    },
    "test": {
        "batch_size": 2048,
        "drop_last": False
    },
    "val": {
        "batch_size": 2048,
        "drop_last": False,
    }
}

if augment_train:
    cfg.dataset.transforms.train = [
        {"name": "ColorJitter", "cfg": {"brightness": 0.7, "contrast": 0.7}},
        {"name": "RandomAffine", "cfg": {"degrees": 6, "translate": (0.2, 0.2), "scale": (0.77, 1.36), "shear": 13}}
    ]

cfg.dataset.transforms.pre_load = [
    {"name": "ToTensor"},
    {"name": "Resize", "cfg": {"size": (28, 28)}},
]

cfg.dataset.transforms.post = [
    {"name": "torch.nn.Flatten", "cfg": {"start_dim": 1}}
]

if permute:
    cfg.dataset.transforms.post.append(
        {"name": "torch_mate.data.transforms.PermuteSequence", "cfg": {"length": 28*28}}
    )

cfg.seed = 4223747124

cfg = cfg.toDict()
