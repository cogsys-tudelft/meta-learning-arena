import random
import pathlib
import pickle
from typing import Tuple, Union, List

import numpy as np

import torch
from torch.utils.data import Subset, ConcatDataset
from torchaudio.datasets import FluentSpeechCommands as FluentSpeechCommandsTorchDataset

from autolightning.types import Phase
from autolightning.dm.few_shot import FewShotMixin

from torch_mate.data.utils import Transformed
from torch_mate.data.utils.get_indices_per_class import get_indices_per_class

from metalarena.autodatasets.lightning_audio_dataset import LightningAudioDataset
from metalarena.autodatasets.fluent_speech_commands import MAX_SAMPLE_LENGTH


MAX_SENTENCE_LENGTH = 10

UNIQUE_WORDS = {'a','allow','anything','audio','bathroom','bedroom','bring',"can't",'change','chinese','cooler','could',"couldn't",'decrease','device','different','down','english','far','fetch','german','get','go','hear','heat','heating','hotter','i','in','increase','is','it',"it's",'juice','kitchen','korean','lamp','language','languages','less','levels','lights','loud','louder','low','lower','main','make','max','me','more','music','mute','my','need','newspaper','now','off','ok','on','open','pause','phone',"phone's",'play','please','practice','put','quiet','quieter','reduce','resume','set','settings','shoes','socks','softer','some','sound','start','stop','switch','system','temperature','that',"that's",'the','this','to','too','turn','up','use','video','volume','washroom','you'}
WORD2IDX = {word: i+1 for i, word in enumerate(sorted(list(UNIQUE_WORDS)))} # +1 to reserve 0 for padding

UNIQUE_TRANSCRIPTIONS = ['Allow a different language', 'Bathroom heat down', 'Bathroom heat up', 'Bathroom lights on', 'Bedroom heat down', 'Bedroom heat up', 'Bedroom lights off', 'Bedroom lights on', 'Bring juice', 'Bring me my shoes', 'Bring me my socks', 'Bring me some juice', 'Bring me the newspaper', 'Bring my shoes', 'Bring my socks', 'Bring newspaper', 'Bring shoes', 'Bring socks', 'Bring some juice', 'Bring the newspaper', 'Change language', 'Change system language', 'Change the language', 'Could you decrease the heating please?', 'Could you decrease the heating?', 'Could you increase the heating please?', 'Could you increase the heating?', 'Decrease audio volume', 'Decrease sound levels', 'Decrease the heating', 'Decrease the heating in the bathroom', 'Decrease the heating in the bedroom', 'Decrease the heating in the kitchen', 'Decrease the heating in the washroom', 'Decrease the temperature', 'Decrease the temperature in the bathroom', 'Decrease the temperature in the bedroom', 'Decrease the temperature in the kitchen', 'Decrease the temperature in the washroom', 'Decrease the volume', 'Decrease volume', 'Far too loud', 'Far too quiet', 'Fetch my shoes', 'Fetch my socks', 'Fetch the newspaper', 'Get me my shoes', 'Get me my socks', 'Get me some juice', 'Get me the newspaper', 'Go get me my shoes', 'Go get me my socks', 'Go get me some juice', 'Go get the newspaper', 'Heat down', 'Heat up', "I can't hear that", "I couldn't hear anything, turn up the volume", 'I need to hear this, increase the volume', 'I need to practice my Chinese. Switch the language', 'I need to practice my English. Switch the language', 'I need to practice my German. Switch the language', 'I need to practice my Korean. Switch the language', 'I need volume', 'Increase the heating', 'Increase the heating in the bathroom', 'Increase the heating in the bedroom', 'Increase the heating in the kitchen', 'Increase the heating in the washroom', 'Increase the sound', 'Increase the sound volume', 'Increase the temperature', 'Increase the temperature in the bathroom', 'Increase the temperature in the bedroom', 'Increase the temperature in the kitchen', 'Increase the temperature in the washroom', 'Increase the volume', 'It’s too loud, turn it down', 'It’s too loud, turn the volume down', 'Kitchen heat down', 'Kitchen heat up', 'Kitchen lights off', 'Kitchen lights on', 'Lamp off', 'Lamp on', 'Language settings', 'Less heat', 'Lights off', 'Lights off in the bedroom', 'Lights off in the kitchen', 'Lights off in the washroom', 'Lights on', 'Lights on in the bathroom', 'Lights on in the bedroom', 'Lights on in the kitchen', 'Lights on in the washroom', 'Louder', 'Louder phone', 'Louder please', 'Lower the volume', 'Make it cooler', 'Make it hotter', 'Make it louder', 'Make it quieter', 'Make the music louder', 'Make the music softer', 'More heat', 'OK now switch the main language to Chinese', 'OK now switch the main language to English', 'OK now switch the main language to German', 'OK now switch the main language to Korean', 'Open language settings', 'Pause', 'Pause music', 'Pause the music', 'Play', 'Play music', 'Play the music', 'Put on the music', 'Quieter', 'Reduce audio volume', 'Resume', 'Resume music', 'Set language to Chinese', 'Set language to English', 'Set language to German', 'Set language to Korean', 'Set my device to Chinese', "Set my phone's language to Chinese", "Set my phone's language to English", "Set my phone's language to German", "Set my phone's language to Korean", 'Set the language', 'Start the music', 'Stop', 'Stop music', 'Stop the music', 'Switch language', 'Switch languages', 'Switch off the lamp', 'Switch off the lights', 'Switch off the lights in the bedroom', 'Switch off the lights in the kitchen', 'Switch off the washroom lights', 'Switch on the bathroom lights', 'Switch on the kitchen lights', 'Switch on the lamp', 'Switch on the lights', 'Switch on the lights in the bedroom', 'Switch on the lights in the kitchen', 'Switch on the washroom lights', 'Switch the bedroom lights off', 'Switch the bedroom lights on', 'Switch the kitchen lights on', 'Switch the language', 'Switch the lights off', 'Switch the lights off in the kitchen', 'Switch the lights on', "That's too quiet", 'That’s too loud', 'This video sound is too low, turn up the volume', 'Too loud', 'Too quiet', 'Turn down the bathroom temperature', 'Turn down the bedroom heat', 'Turn down the heat', 'Turn down the heat in the bathroom', 'Turn down the heat in the kitchen', 'Turn down the heat in the washroom', 'Turn down the temperature', 'Turn down the temperature in the bedroom', 'Turn down the temperature in the kitchen', 'Turn down the volume', 'Turn down the washroom temperature', 'Turn it down', 'Turn it up', 'Turn off the kitchen lights', 'Turn off the lamp', 'Turn off the lights', 'Turn off the lights in the bedroom', 'Turn off the music', 'Turn off the washroom lights', 'Turn on the bathroom lights', 'Turn on the kitchen lights', 'Turn on the lamp', 'Turn on the lights', 'Turn on the lights in the bedroom', 'Turn on the washroom lights', 'Turn sound down', 'Turn sound up', 'Turn the bathroom lights on', 'Turn the bedroom heat down', 'Turn the bedroom heat up', 'Turn the bedroom lights off', 'Turn the bedroom lights on', 'Turn the heat down', 'Turn the heat down in the bathroom', 'Turn the heat down in the kitchen', 'Turn the heat down in the washroom', 'Turn the heat up', 'Turn the heat up in the bathroom', 'Turn the heat up in the kitchen', 'Turn the heat up in the washroom', 'Turn the kitchen lights on', 'Turn the kitchen temperature down', 'Turn the kitchen temperature up', 'Turn the lamp off', 'Turn the lamp on', 'Turn the lights off', 'Turn the lights off in the kitchen', 'Turn the lights on', 'Turn the lights on in the kitchen', 'Turn the sound up', 'Turn the temperature down', 'Turn the temperature down in the bathroom', 'Turn the temperature down in the washroom', 'Turn the temperature in the bedroom down', 'Turn the temperature in the bedroom up', 'Turn the temperature up', 'Turn the temperature up in the bathroom', 'Turn the temperature up in the washroom', 'Turn the volume down', 'Turn the volume up', 'Turn the washroom lights off', 'Turn the washroom lights on', 'Turn up the bathroom temperature', 'Turn up the bedroom heat', 'Turn up the heat', 'Turn up the heat in the bathroom', 'Turn up the heat in the kitchen', 'Turn up the heat in the washroom', 'Turn up the temperature', 'Turn up the temperature in the bedroom', 'Turn up the temperature in the kitchen', 'Turn up the volume', 'Turn up the washroom temperature', 'Turn volume down', 'Turn volume up', 'Use a different language', 'Volume down', 'Volume lower', 'Volume max', 'Volume mute', 'Volume up', 'Washroom heat down', 'Washroom heat up', 'Washroom lights off', 'Washroom lights on']
TRANSCRIPTION2IDX = {transcription: i for i, transcription in enumerate(UNIQUE_TRANSCRIPTIONS)}
IDX2TRANSCRIPTION = {i: transcription for transcription, i in TRANSCRIPTION2IDX.items()}


def tokenize_transcription(transcription: str):
    transcription = transcription.replace(".", " ").replace("?", "").replace(",", "")

    while "  " in transcription:
        transcription = transcription.replace("  ", " ")

    words = transcription.lower().split(" ")

    tokenized = [WORD2IDX[word.replace("’", "'")] for word in words]

    # Pad to max length
    tokenized = [0] * (MAX_SENTENCE_LENGTH - len(tokenized)) + tokenized

    return torch.tensor(tokenized, dtype=torch.long)


def zero_shot_fc_transform(entry: Tuple[torch.Tensor, int, str, int, str, str, str, str]):
    # entry = [waveform, sample_rate, file_name, speaker_id, transcription, action, object, location]

    waveform = entry[0]

    if waveform.shape[1] != MAX_SAMPLE_LENGTH:
        pad_length = MAX_SAMPLE_LENGTH - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (pad_length, 0))  # pad the start of the audio until 1 second

    transcription = entry[4]
    tokenized_transcription = tokenize_transcription(transcription)
    label = TRANSCRIPTION2IDX[transcription]

    # first return X_train, then X_test, then y for compatibility with few-shot mixin from AutoLightning
    return tokenized_transcription, waveform, label


def split_in_groups(data: List[int], splits: Union[Tuple[float, float, float], Tuple[int, int, int]]):
    n = len(data)

    split_percentages = None

    if all(isinstance(split, float) for split in splits):
        split_percentages = splits
    elif all(isinstance(split, int) for split in splits):
        split_percentages = (splits[0] / n, splits[1] / n, splits[2] / n)
    else:
        raise ValueError("splits should be all float or all int")

    idx1 = round(n * split_percentages[0])
    idx2 = idx1 + round(n * split_percentages[1])

    return {
        'train': data[:idx1],
        'valid': data[idx1:idx2],
        'test': data[idx2:]
    }


class ZeroShotFluentSpeechCommands(FewShotMixin, LightningAudioDataset):
    def __init__(self,
                 root: str,
                 train_test_val_split: Union[Tuple[float, float, float], Tuple[int, int, int]] = (0.75, 0.1, 0.15),
                 random_classes_per_split: bool = False,
                 **kwargs):
        self.root = root
        self.train_test_val_split = train_test_val_split
        self.random_classes_per_split = random_classes_per_split

        assert sum(train_test_val_split) == 1.0 or sum(train_test_val_split) == 248, "train_test_val_split should sum to 1.0 (fractions) or 248 (absolute number of classes)"

        super().__init__(target_batch_transforms=None, samples_per_class=None, **kwargs)

    def setup(self, stage: str):
        all_class_idxs = list(range(len(UNIQUE_TRANSCRIPTIONS)))

        if self.random_classes_per_split:
            random.seed(self.seed)
            random.shuffle(all_class_idxs)
        
        classes_per_split = split_in_groups(all_class_idxs, self.train_test_val_split)

        indices_dir = pathlib.Path(__file__).parents[0] / ".."/ "artifacts"

        indices_per_split = {'train': [], 'valid': [], 'test': []}
        concat_class_mapping = []
        og_datasets = []

        phases = ['train', 'valid', 'test']

        for phase in phases:
            with open(indices_dir / f"flsc_{phase}_per_class_indices.pkl", "rb") as f:
                og_class_indices_map = pickle.load(f)

            start_idx = sum(len(og_datasets[i]) for i in range(len(og_datasets)))

            # Assign the classes in this split to the new meta/train/test/val splits
            for assign_phase in phases:
                for class_idx in classes_per_split[assign_phase]:
                    indices_per_split[assign_phase].extend(list(np.array(og_class_indices_map[class_idx], dtype=np.int64) + start_idx))

            og_datasets.append(FluentSpeechCommandsTorchDataset(root=self.root, subset=phase))

            # For all samples in this dataset, find the class idx and store it in concat_class_mapping
            for i in range(len(og_datasets[-1])):
                # find the class idx for this sample
                for class_idx, indices in og_class_indices_map.items():
                    if i in indices:
                        concat_class_mapping.append((class_idx,))
                        break

        assert set(classes_per_split['train']).isdisjoint(set(classes_per_split['valid']))
        assert set(classes_per_split['train']).isdisjoint(set(classes_per_split['test']))
        assert set(classes_per_split['valid']).isdisjoint(set(classes_per_split['test']))

        assert set(indices_per_split['train']).isdisjoint(set(indices_per_split['valid']))
        assert set(indices_per_split['train']).isdisjoint(set(indices_per_split['test']))
        assert set(indices_per_split['valid']).isdisjoint(set(indices_per_split['test']))

        assert len(indices_per_split['train']) + len(indices_per_split['valid']) + len(indices_per_split['test']) == sum(len(og_dataset) for og_dataset in og_datasets)

        full_og_dataset = ConcatDataset(og_datasets)
        
        if stage == 'fit' or stage == 'validate':
            key = 'valid'

            self.val_set = Subset(full_og_dataset, indices_per_split[key])
            self.samples_per_class = {key[:3]: get_indices_per_class(Subset(concat_class_mapping, indices_per_split[key]))}

            if stage == 'fit':
                key = 'train'

                self.train_set = Subset(full_og_dataset, indices_per_split[key])
                self.samples_per_class[key] = get_indices_per_class(Subset(concat_class_mapping, indices_per_split[key]))
        elif stage == 'test':
            self.test_set = Subset(full_og_dataset, indices_per_split[stage])
            self.samples_per_class = {stage: get_indices_per_class(Subset(concat_class_mapping, indices_per_split[stage]))}
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def get_dataset(self, phase: Phase):
        # Samples in train, val and test splits are already distributed in a similar way, so no need to balance them
        return Transformed(
            getattr(self, f"{phase}_set"),
            transform=zero_shot_fc_transform
        )
