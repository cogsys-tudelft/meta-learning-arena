from typing import Tuple

import torch

from autolightning.types import Phase

from torch_mate.data.utils import Transformed

from metalarena.autodatasets.lightning_audio_dataset import LightningAudioDataset


UNIQUE_INTENT_SLOTS = [('activate', 'lamp', 'none'),
 ('activate', 'lights', 'bedroom'),
 ('activate', 'lights', 'kitchen'),
 ('activate', 'lights', 'none'),
 ('activate', 'lights', 'washroom'),
 ('activate', 'music', 'none'),
 ('bring', 'juice', 'none'),
 ('bring', 'newspaper', 'none'),
 ('bring', 'shoes', 'none'),
 ('bring', 'socks', 'none'),
 ('change language', 'Chinese', 'none'),
 ('change language', 'English', 'none'),
 ('change language', 'German', 'none'),
 ('change language', 'Korean', 'none'),
 ('change language', 'none', 'none'),
 ('deactivate', 'lamp', 'none'),
 ('deactivate', 'lights', 'bedroom'),
 ('deactivate', 'lights', 'kitchen'),
 ('deactivate', 'lights', 'none'),
 ('deactivate', 'lights', 'washroom'),
 ('deactivate', 'music', 'none'),
 ('decrease', 'heat', 'bedroom'),
 ('decrease', 'heat', 'kitchen'),
 ('decrease', 'heat', 'none'),
 ('decrease', 'heat', 'washroom'),
 ('decrease', 'volume', 'none'),
 ('increase', 'heat', 'bedroom'),
 ('increase', 'heat', 'kitchen'),
 ('increase', 'heat', 'none'),
 ('increase', 'heat', 'washroom'),
 ('increase', 'volume', 'none')]

INTENT2IDX = {intent_slots: i for i, intent_slots in enumerate(UNIQUE_INTENT_SLOTS)}

UNIQUE_VALUES_PER_SLOT = [set() for _ in range(3)]

for intent_tuple in UNIQUE_INTENT_SLOTS:
    UNIQUE_VALUES_PER_SLOT[0].add(intent_tuple[0])
    UNIQUE_VALUES_PER_SLOT[1].add(intent_tuple[1])
    UNIQUE_VALUES_PER_SLOT[2].add(intent_tuple[2])

SLOTS2IDXS = [{value: i for i, value in enumerate(sorted(list(slot_values)))} for slot_values in UNIQUE_VALUES_PER_SLOT]

MAX_SAMPLE_LENGTH = 211627  # 13.22 seconds at 16kHz


class FSCTransform:
    def __init__(self, per_slot_targets: bool = False):
        self.per_slot_targets = per_slot_targets

    def __call__(self, entry: Tuple[torch.Tensor, int, str, int, str, str, str, str]):
        # entry = [waveform, sample_rate, file_name, speaker_id, transcription, action, object, location]

        waveform = entry[0]

        if waveform.shape[1] != MAX_SAMPLE_LENGTH:
            pad_length = MAX_SAMPLE_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (pad_length, 0))  # pad the start of the audio until 1 second

        intent_slots = entry[-3:]

        if self.per_slot_targets:
            label = tuple([SLOTS2IDXS[i][intent_slots[i]] for i in range(3)])
        else:
            label = INTENT2IDX[intent_slots]

        return waveform, label


class FluentSpeechCommands(LightningAudioDataset):
    def __init__(self,
                 root: str,
                 per_slot_targets: bool = False,
                 **kwargs):
        super().__init__(
            dataset={
                "class_name": "torchaudio.datasets.FluentSpeechCommands",
                "args": dict(
                    defaults=dict(
                        root=root
                    ),
                    train=dict(subset='train'),
                    val=dict(subset='valid'),
                    test=dict(subset='test')
                )
            },
            random_split=None,
            requires_prepare=False,
            target_batch_transforms=None,
            **kwargs
        )

        self.per_slot_targets = per_slot_targets

    def get_dataset(self, phase: Phase):
        # Samples in train, val and test splits are already distributed in a similar way, so no need to balance them
        return Transformed(
            super().get_dataset(phase),
            transform=FSCTransform(per_slot_targets=self.per_slot_targets)
        )
