from typing import Tuple

import torch

from autolightning.types import Phase

from torch_mate.data.utils import Transformed

from metalarena.autodatasets.lightning_audio_dataset import LightningAudioDataset


SAMPLE_RATE = 16000
WORDS = ['right','eight','cat','tree','backward','learn','bed','happy','go','dog','no','wow','follow','nine','left','stop','three','sheila','one','bird','zero','seven','up','visual','marvin','two','house','down','six','yes','on','five','forward','off','four']
WORD2IDX = {word: i for i, word in enumerate(WORDS)}

TAR_NAME = 'speech_commands_v0.02'
FOLDER_NAME = 'SpeechCommands'


def sc35tf(entry: Tuple[torch.Tensor, int, str, str, int]):
    # entry = [waveform, sample_rate, label, speaker_id, utterance_number]

    waveform = entry[0]
    int_label = WORD2IDX[entry[2]]

    if waveform.shape[1] != SAMPLE_RATE:
        pad_length = SAMPLE_RATE-waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (pad_length, 0)) # pad the start of the audio until 1 second

    return waveform, int_label


class SpeechCommandsV2_35C(LightningAudioDataset):
    def __init__(self,
                 root: str,
                 download: bool = False,
                 take_log: bool = False,
                 **kwargs):
        super().__init__(
            dataset={
                "class_name": "torchaudio.datasets.SPEECHCOMMANDS",
                "args": dict(
                    defaults=dict(
                        root=root,
                        url=TAR_NAME,
                        folder_in_archive=FOLDER_NAME,
                        download=download
                    ),
                    train=dict(subset='training'),
                    val=dict(subset='validation'),
                    test=dict(subset='testing')
                )
            },
            random_split=None,
            requires_prepare=download,
            take_log=take_log,
            target_batch_transforms=None,
            **kwargs
        )

    def get_dataset(self, phase: Phase):
        return Transformed(
            super().get_dataset(phase),
            transform=sc35tf
        )
