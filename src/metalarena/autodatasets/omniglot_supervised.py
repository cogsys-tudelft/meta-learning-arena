from typing import Optional, List, Literal, Union, Tuple
import pickle
import pathlib
from functools import partial

import torch
from torch.utils.data import random_split, Subset, Dataset
import torchvision

from torch_mate.data.utils import RotationExtended, Transformed

from autolightning import AutoDataModule


OmniglotVariantType = Literal["Vinyals", "MAML", "Lake"]
OmniglotApproachType = Literal["supervised", "few-shot", "both"]


SAMPLES_PER_CLASS = 20
TOTAL_CLASSES = 1623
NUM_BACKGROUND_CLASSES = 964
VINYALS_CLASS_VAL_PERCENTAGE = 172/1200


def _make_labels_start_from_zero(dataset: Dataset):
    """Make sure all train set indices are in the range of the number of train classes
    """

    classes = list(set([dataset[i][1] for i in range(len(dataset))]))
    return Transformed(dataset, target_transform=lambda i: classes.index(i))


def _get_train_val_indices(samples_per_class: int, classes: torch.Tensor):
    indices = []

    for i in classes:
        indices.extend([i * samples_per_class + j for j in range(samples_per_class)])

    return indices


def _get_variant_split_indices(variant: str, class_val_percentage: float = VINYALS_CLASS_VAL_PERCENTAGE, shuffle_classes: bool = False):
    if shuffle_classes and variant != "MAML":
        raise ValueError(f"Cannot shuffle classes for split {variant}")

    total_train_val_classes = 964 if variant == 'Lake' else 1200

    # By default, we use the Vinyals split (1028 train, 172 val, rest (423) test)
    num_train_classes = int(total_train_val_classes * (1 - class_val_percentage))
    num_val_classes = total_train_val_classes - num_train_classes
    num_test_classes = TOTAL_CLASSES - total_train_val_classes

    if variant == "Vinyals":
        # As used by Vinyals et al. in their Matching Networks paper and as used by Snell et al. in their Prototypical Networks paper
        # See their code here: https://github.com/jakesnell/prototypical-networks/tree/master/data/omniglot/splits/vinyals
        assert num_train_classes == 1028, "Vinyals split only supports a validation percentage of 172/1200 as per the official implementation"

        with open(pathlib.Path(__file__).parents[0] / ".."/ "artifacts" / "omniglot_indices_vinyals_split.pkl", 'rb') as f:
            train_indices, val_indices, test_indices = pickle.load(f)
    elif variant == "MAML":
        # Randomly select train_classes and val_classes from the total set of classes
        # as per the official MAML implementation of Omniglot:
        # https://github.com/cbfinn/maml/blob/a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49/main.py#L267-L282 
        # See also: https://stats.stackexchange.com/questions/592229/why-does-the-maml-split-the-omniglot-data-set-randomly-on-every-run
        shuffled_indices = torch.randperm(TOTAL_CLASSES) if shuffle_classes == True else torch.arange(TOTAL_CLASSES)

        train_classes = shuffled_indices[:num_train_classes]
        val_classes = shuffled_indices[num_train_classes:num_train_classes + num_val_classes]
        test_classes = shuffled_indices[-num_test_classes:]

        # Get the indices of the samples belonging to the selected classes
        train_indices, val_indices, test_indices = [_get_train_val_indices(SAMPLES_PER_CLASS, classes) for classes in [train_classes, val_classes, test_classes]]
    elif variant == "Lake":
        train_indices = list(range(num_train_classes*SAMPLES_PER_CLASS))
        val_indices = list(range(num_train_classes*SAMPLES_PER_CLASS, (num_train_classes + num_val_classes)*SAMPLES_PER_CLASS))
        test_indices = list(range(total_train_val_classes*SAMPLES_PER_CLASS, TOTAL_CLASSES*SAMPLES_PER_CLASS))
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return (train_indices, val_indices, test_indices), num_train_classes


def add(x: int, amount: int):
    return x + amount


class OmniglotVariants:
    def __init__(self,
                 root: Union[str, pathlib.Path],
                 download: bool = False,
                 variant: OmniglotVariantType = "Vinyals",
                 class_val_percentage: float = VINYALS_CLASS_VAL_PERCENTAGE,
                 val_percentage: float = 0.1,
                 shuffle_classes: bool = False,
                 rotations: Optional[List[int]] = None,
                 rotate_test_classes: bool = False):
        self.root = root
        self.download = download
        
        self.variant = variant
        self.class_val_percentage = class_val_percentage
        self.val_percentage = val_percentage
        self.shuffle_classes = shuffle_classes
        self.rotations = rotations
        self.rotate_test_classes = rotate_test_classes

    def split(self, approach: OmniglotApproachType = "supervised") -> Union[Tuple[Tuple[Subset, Subset], int], Tuple[Tuple[Subset, Subset, Subset], int], Tuple[Tuple[Subset, Subset], Tuple[Subset, Subset, Subset], int]]:
        if approach not in ["supervised", "few-shot", "both"]:
            raise ValueError(f"Unknown approach: {approach}")

        indices_per_split, num_train_classes = _get_variant_split_indices(self.variant, self.class_val_percentage, self.shuffle_classes)

        background_set = torchvision.datasets.Omniglot(root=self.root, download=self.download, background=True)

        # Currently, the background set contains classes labeled 0-963 and the non-background set contains classes labeled 0-659 while
        # these are not the classes. Therefore, we add 964 to the non-background set labels to make the labels unique to the classes.
        non_background_set = torchvision.datasets.Omniglot(root=self.root, download=self.download, background=False, target_transform=partial(add, amount=NUM_BACKGROUND_CLASSES))
        all_splits_data = background_set + non_background_set

        train_set_classes, val_set_classes, test_set_classes = [Subset(all_splits_data, indices) for indices in indices_per_split]

        train_set_classes = _make_labels_start_from_zero(train_set_classes)

        rotate = lambda x: RotationExtended(x, SAMPLES_PER_CLASS, self.rotations)

        if self.rotations:
            train_set_classes = rotate(train_set_classes)
        
        data_to_return = []

        if approach == "supervised" or approach == "both":
            if self.rotate_test_classes == True:
                raise ValueError("Cannot rotate test classes for supervised learning")
            
            generator = torch.Generator().manual_seed(41)
            train_set, val_set = random_split(train_set_classes, [1 - self.val_percentage, self.val_percentage], generator=generator)

            data_to_return.append((train_set, val_set))

        if approach == "few-shot" or approach == "both":
            # Make sure all data set indices are in the range of the number of train classes
            val_set_classes = _make_labels_start_from_zero(val_set_classes)
            test_set_classes = _make_labels_start_from_zero(test_set_classes)

            if self.rotations:
                val_set_classes = rotate(val_set_classes)

                if self.rotate_test_classes:
                    test_set_classes = rotate(test_set_classes)

            data_to_return.append((train_set_classes, val_set_classes, test_set_classes))

        data_to_return.append(num_train_classes)

        return tuple(data_to_return)


class OmniglotSupervised(AutoDataModule):
    def __init__(self,
                 root: str,
                 download: bool = False,
                 class_val_percentage: float = 172/1200,
                 val_percentage: float = 0.1,
                 variant: Literal["Vinyals", "MAML", "Lake"] = "Vinyals",
                 shuffle_classes: bool = False,
                 rotate_test_classes: bool = False,
                 rotations: Optional[List[int]] = None,
                 **kwargs):
        super().__init__(
            dataset=None,
            random_split=None,
            requires_prepare=False,
            **kwargs
        )

        self.root = root
        self.download = download

        self.class_val_percentage = class_val_percentage
        self.val_percentage = val_percentage
        self.variant = variant
        self.shuffle_classes = shuffle_classes
        self.rotations = rotations

        self.rotate_test_classes = rotate_test_classes

    def get_omniglot_variant(self, approach: OmniglotApproachType = "supervised"):
        return OmniglotVariants(
            root=self.root,
            download=self.download,
            variant=self.variant,
            class_val_percentage=self.class_val_percentage,
            val_percentage=self.val_percentage,
            shuffle_classes=self.shuffle_classes,
            rotations=self.rotations,
            rotate_test_classes=self.rotate_test_classes
        ).split(approach=approach)

    def prepare_data(self):
        if self.download:
            self.get_omniglot_variant()

    def setup(self, stage: str):
        if stage != "fit":
            raise ValueError(f"Unsupported stage: {stage}")

        (self.train_set, self.val_set), _ = self.get_omniglot_variant()

    def get_dataset(self, phase: str):
        if phase == 'train':
            return self.train_set
        elif phase == 'val':
            return self.val_set
        
        raise ValueError(f"Unsupported phase: {phase}")
