"""Pick the least-used GPU (LUG) from the GPUs that we have access to."""

from torch_mate.utils.gpu_picker_utils import sort_gpus_by_memory_usage

config = {"trainer": {"devices": sort_gpus_by_memory_usage([2,3,6,7])[:1] }}
