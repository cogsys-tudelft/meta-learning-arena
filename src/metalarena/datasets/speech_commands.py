import os
from typing import Union
from pathlib import Path

from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.utils import _load_waveform

from AudioLoader.speech.speechcommands import name2idx


SAMPLE_RATE = 16000
URL = "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
FOLDER_IN_ARCHIVE = "SpeechCommands/speech_commands_test_set_v0.02"
SILENCE_LABEL = name2idx["_silence_"]


class SPEECHCOMMANDS_12C_TEST(SPEECHCOMMANDS):
    def __init__(self, root: Union[str, Path], download: bool = False):
        """Note: this dataset has no overlap at all with the training split of
        AudioLoader.speech.speechcommands.SPEECHCOMMANDS."""

        super().__init__(root, URL, FOLDER_IN_ARCHIVE, download)
        self._silence = self._load_silence_folder(os.path.join(self._path, '_silence_'))

    def _load_silence_folder(self, folder_path):
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

    def __getitem__(self, n):
        if n < len(self._silence):
            return _load_waveform(".", self._silence[n], SAMPLE_RATE), SILENCE_LABEL
    
        entry = super().__getitem__(n - len(self._silence))

        return entry[0], name2idx[entry[2]]
    
    def __len__(self):
        return super().__len__() + len(self._silence)
