from autolightning import cc


LABEL_EMBEDDER_TYPE = "onehot" # can be onehot, fixed-embedder, learnable-embedder, full-context
LABEL_EMBEDDING_DIM = 32
EMBEDDER_OUT_DIM = 48
FULL_CONTEXT_TCN_DIM = 32
SHUFFLE_LABELS = True
USE_PRETRAINED_EMBEDDER = True
DISABLE_EMBEDDER_GRAD = True
PRE_TRAINED_EMBEDDER_PATH = "/space/ddenblanken/Projects/meta-learning-arena-new/test_wandb_lightning/vg3yq0yt/checkpoints/epoch=0-step=60000.ckpt"
WAYS = 5


mlp_in_dim = (WAYS + 1) * EMBEDDER_OUT_DIM
input_length = None

if LABEL_EMBEDDER_TYPE == "onehot":
    mlp_in_dim += WAYS * WAYS

    label_embedder = cc("torch_mate.nn.OneHot", num_classes=WAYS)
elif LABEL_EMBEDDER_TYPE == "learnable-embedder" or LABEL_EMBEDDER_TYPE == "fixed-embedder":
    mlp_in_dim += WAYS * LABEL_EMBEDDING_DIM

    label_embedder = cc("torch.nn.Embedding", num_embeddings=WAYS, embedding_dim=LABEL_EMBEDDING_DIM)

    if LABEL_EMBEDDER_TYPE == "fixed-embedder":
        label_embedder = cc("autolightning.disable_grad", module=label_embedder)
elif LABEL_EMBEDDER_TYPE == "full-context":
    input_length = 784 * (WAYS + 1) + WAYS * LABEL_EMBEDDING_DIM
    label_embedder = None
else:
    raise ValueError(f"Unknown label embedder type: {LABEL_EMBEDDER_TYPE}")

sample_embedder_module = cc(
    "tcn_lib.TCN",
    input_size=1,
    channel_sizes=FULL_CONTEXT_TCN_DIM if LABEL_EMBEDDER_TYPE == "full-context" else [32, 32, 32, 48, 48, 48, EMBEDDER_OUT_DIM],
    input_length=input_length,
    output_size=WAYS if LABEL_EMBEDDER_TYPE == "full-context" else -1,
    kernel_size=5,
    dropout=0.025,
    batch_norm=True,
    residual=True,
    weight_norm=False,
    zero_init_residual=False
)

if USE_PRETRAINED_EMBEDDER:
    sample_embedder_module = cc("autolightning.load", file_path=PRE_TRAINED_EMBEDDER_PATH, state_dict_submodule="net._orig_mod", module=sample_embedder_module)

    if DISABLE_EMBEDDER_GRAD:
        sample_embedder_module = cc("autolightning.disable_grad", module=sample_embedder_module)
elif DISABLE_EMBEDDER_GRAD:
    raise ValueError("Cannot disable grad for sample embedder if not using pretrained embedder")

config = {
  "model": {
    "class_path": "autolightning.lm.ICLClassifier",
    "init_args": {
      "combine_batch_and_samples": True,
      "net": {
        "class_path": "autolightning.compile",
        "init_args": {
          "compiler_path": "torch.compile",
          "module": {
            "class_path": "torchvision.ops.MLP",
            "init_args": {
              "in_channels": mlp_in_dim,
              "hidden_channels": [96, 96, WAYS]
            }
          }
        }
      },
      "label_embedder": label_embedder,
      "sample_embedder": {
        "class_path": "autolightning.compile",
        "init_args": {
          "compiler_path": "torch.compile",
          "module": sample_embedder_module
        }
      }
    }
  },
  "data": {
    "class_path": "metalarena.autodatasets.OmniglotFewShot",
    "init_args": {
      "root": "data/",
      "download": True,
      "pre_load": True,
      "variant": "Vinyals",
      "rotations": [0, 90, 180, 270],
      "query_ways": 1,
      "ways": WAYS,
      "shuffle_classes": False,
      "rotate_test_classes": True,
      "keep_original_labels": False,
      "shuffle_labels": SHUFFLE_LABELS,
      "transforms": {
        "pre_load": [
          cc("torchvision.transforms.Resize", size=[28, 28]),
          cc("torchvision.transforms.ToTensor")
        ],
        "post": [cc("torch.nn.Flatten", start_dim=1)]
      },
      "dataloaders": {
        "batch_size": 512,
        "num_workers": 16,
        "prefetch_factor": 8,
        "persistent_workers": True,
        "pin_memory": True
      }
    }
  },
  "optimizer": cc("torch.optim.Adam", lr=1e-3),
  "seed_everything": 4223747124,
  "trainer": {
    "max_steps": 120000,
    "val_check_interval": 1000,
    "limit_val_batches": 50
  },
}
