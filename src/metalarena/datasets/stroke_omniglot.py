"""Code heavily inspired by:

https://github.com/nerdimite/maml/blob/master/dataset.py.
"""

import csv
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def read_stroke(file_path: str):
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        data = []

        for i, row in enumerate(reader):
            # Skip header that says 'START'
            # Also ignore 'BREAK' rows, indicating the start of a new stroke
            if i == 0 or row[0] == "BREAK":
                continue

            data.append((float(row[0]), float(row[1]), int(row[2])))

    return data


def load_characters(root: str, alphabet: str):
    """Loads the characters from a given alphabet.

    Args:
        root (str): Root directory
        alphabet (str): Folder name of alphabet
    Returns:
        (tuple) of:
            (list): images
            (list): labels
    """
    X = []
    y = []

    alphabet_path = os.path.join(root, alphabet)
    characters = os.listdir(alphabet_path)

    for char in characters:
        char_path = os.path.join(alphabet_path, char)
        images = os.listdir(char_path)

        for img in images:
            image = read_stroke(os.path.join(char_path, img))

            X.append(image)
            y.append(f'{alphabet}_{char}')

    return X, y


def load_data(root: str):
    """Loads the full Omniglot dataset from a root directory.

    Args:
        root (str): Path of Omniglot dataset

    Returns:
        (tuple) of:
            (ndarray): images
            (ndarray): labels
    """

    X_data = []
    y_data = []

    alphabets = os.listdir(root)

    for alphabet in alphabets:
        X, y = load_characters(root, alphabet)

        X_data.extend(X)
        y_data.extend(y)

    # Find longest sequence, largest and smallest x and y values
    max_len = 0
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    for i, seq in enumerate(X_data):
        np_seq = np.array(seq)

        X_data[i] = np_seq

        if len(seq) > max_len:
            max_len = len(seq)

        curr_min_x = np_seq[:, 0].min()
        curr_max_x = np_seq[:, 0].max()
        curr_min_y = np_seq[:, 1].min()
        curr_max_y = np_seq[:, 1].max()

        if curr_max_x > max_x:
            max_x = curr_max_x

        if curr_min_x < min_x:
            min_x = curr_min_x

        if curr_max_y > max_y:
            max_y = curr_max_y

        if curr_min_y < min_y:
            min_y = curr_min_y

    print(f'Max length: {max_len}')
    print(f'Min x: {min_x}')
    print(f'Max x: {max_x}')
    print(f'Min y: {min_y}')
    print(f'Max y: {max_y}')

    # Normalize all sequences and pad to max_len with zeros on the left
    for i, seq in enumerate(X_data):
        X_data[i][:, 0] = (seq[:, 0] - min_x) / (max_x - min_x)
        X_data[i][:, 1] = (seq[:, 1] - min_y) / (max_y - min_y)
        X_data[i] = np.pad(seq, ((max_len - len(seq), 0), (0, 0)), 'constant')

    X_data = torch.tensor(np.array(X_data), dtype=torch.float32)
    y_data = np.array(y_data)

    return X_data, y_data


class StrokeOmniglot(Dataset):

    def __init__(self,
                 root: str,
                 remove_timestamps: bool = True,
                 transform: Optional[Callable] = None):
        self.root = root
        self.remove_timestamps = remove_timestamps
        self.transform = transform

        self.X, self.y = load_data(root)
        self.classes = np.unique(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        """Get item at index idx.

        Args:
            idx (int): Index of item to get

        Returns:
            (torch.Tensor, int): Tuple of data, shape = (sequence length, #channels) and label
        """
        data = self.X[idx]

        if self.remove_timestamps:
            # Take first, second and fourth column
            data = data[:, [0, 1]]

        # Map string label to index from len(self.classes)
        label = np.where(self.classes == self.y[idx])[0][0]

        if self.transform:
            data = self.transform(data)

        return data, label
