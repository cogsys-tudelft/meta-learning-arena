import math
import random

from torch.utils.data import Dataset, Subset

from autolightning.types import Phase, DatasetType

from torch_mate.data.transforms import AddNoise
from torch_mate.data.utils import LabelDependentTransformed
from torch_mate.data.samplers import ImbalancedClassSampler
from torch_mate.data.utils.get_class_counts import get_class_counts

from metalarena.datasets.speech_commands import SPEECHCOMMANDS_12C_TEST
from metalarena.autodatasets.lightning_audio_dataset import LightningAudioDataset


NOISE_LABEL = 10
UNKNOWN_LABEL = 11

SPECIAL_LABELS = [NOISE_LABEL, UNKNOWN_LABEL]

TAR_NAME = 'speech_commands_v0.02'
FOLDER_NAME = 'SpeechCommands'

# Must be a named module-level function (not a lambda) so it can be pickled
# when DataLoader workers are spawned (default start method on macOS)
def TUPLE2LABEL(_0, label, _2, _3, _4):
    return label


def _calculate_balanced_data_set_size(dataset: Dataset) -> int:
    class_counts = get_class_counts(dataset)

    # Take into account that noise classes and unknown classes can potentially not be present, for example in the 11-class variant
    known_keywords_count = (sum(class_counts.values()) - class_counts.get(NOISE_LABEL, 0) - class_counts.get(UNKNOWN_LABEL, 0))

    return round(known_keywords_count / 10 * (10 + (NOISE_LABEL in class_counts) + (UNKNOWN_LABEL in class_counts)))


class SpeechCommandsV2_12C(LightningAudioDataset):
    def __init__(self,
                 root: str,
                 download: bool = False,
                 take_log: bool = False,
                 noise_p: float = 0.0,
                 max_noise_level: float = 0.0,
                 random_val_silence_unknown_samples: bool = False,
                 **kwargs):
        super().__init__(
            dataset={
                "class_name": "AudioLoader.speech.SPEECHCOMMANDS_12C",
                "args": dict(
                    defaults=dict(
                        root=root,
                        url=TAR_NAME,
                        folder_in_archive=FOLDER_NAME,
                        download=download,
                        target_transform=TUPLE2LABEL,
                        subset='training'
                    ),
                    train=dict(subset='training'),
                    val=dict(subset='validation'),
                    test=dict(subset='testing')
                )
            },
            random_split=None,
            # This dataseset loads itself in cache when instantiated,
            # so doing that during the prepare step as well takes extra time
            requires_prepare=False,
            take_log=take_log,
            target_batch_transforms=None,
            **kwargs
        )

        self.root = root
        self.download = download

        self.noise_p = noise_p
        self.max_noise_level = max_noise_level
        self.random_val_silence_unknown_samples = random_val_silence_unknown_samples

    def setup(self, stage: str):      
        super().setup(stage)
      
        if stage == 'fit':
            train_set = self.get_dataset("train")
            balanced_set_count = _calculate_balanced_data_set_size(train_set)
            self._train_sampler = ImbalancedClassSampler(train_set, balanced_set_count)

    def get_dataset(self, phase: Phase):
        if phase == 'test':
            ds = SPEECHCOMMANDS_12C_TEST(self.root, download=self.download)

            return ds

        ds = super().get_dataset(phase)

        # Make sure that data distribution of the validation set is the same as the test set
        if phase == 'val':
            counts = get_class_counts(ds)

            # Dont include silence and unknown in the count
            instruction_kw_count = sum(counts[k] for k in counts if k not in SPECIAL_LABELS)
            # Get the indices of the samples that are not silence and unknown
            subset_indices = [i for i in range(len(ds)) if ds[i][1] not in SPECIAL_LABELS]

            for special_label in [10, 11]:
                indices = [i for i in range(len(ds)) if ds[i][1] == special_label]

                # Add 10% of unknown and silence samples
                # Use math.ceil, in the same way as the original implementation:
                # https://github.com/tensorflow/tensorflow/blob/3c75664e72c40fc202fd986903cea39bd526f63d/tensorflow/examples/speech_commands/input_data.py#L318
                keep_count = math.ceil(instruction_kw_count / 10)

                if keep_count < len(indices):
                    if self.random_val_silence_unknown_samples:
                        indices = random.sample(indices, keep_count)
                    else:
                        indices = indices[:keep_count]
                else:
                    if self.random_val_silence_unknown_samples:
                        indices = random.choices(indices, k=keep_count)
                    else:
                        # Need to oversample to meet the required count
                        oversample_count = keep_count - len(indices)
                        oversampled_indices = indices * (oversample_count // len(indices))
                        oversampled_indices += indices[:oversample_count % len(indices)]

                        indices = indices + oversampled_indices

                subset_indices.extend(indices)

            ds = Subset(ds, subset_indices)

        return ds
        
    def get_dataloader_kwargs(self, phase: Phase, dataset: DatasetType):
        kwargs = super().get_dataloader_kwargs(phase, dataset)

        if phase == 'train':
            assert 'sampler' not in kwargs, f"Sampler is not supported for this autodataset during {phase}, as a balanced sampler is used instead"
            kwargs['sampler'] = self._train_sampler if phase == 'train' else self._val_sampler

        return kwargs
    
    def get_transformed_dataset(self, phase: Phase):
        dataset = super().get_transformed_dataset(phase)

        # Noise should be added after applying time shift
        if phase == 'train' and self.noise_p > 0:
            # Sample from the training set data without transformations
            add_noise_augmentation = AddNoise([data for (data, label) in dataset if label == NOISE_LABEL], [NOISE_LABEL], p=self.noise_p, max_noise_level=self.max_noise_level)

            return LabelDependentTransformed(dataset, add_noise_augmentation)

        return dataset
