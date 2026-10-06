from typing import Optional

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import IterableDataset


class AddingProblem(IterableDataset):

    def __init__(self,
                 N: Optional[int] = None,
                 seq_len=6,
                 high=1,
                 number_of_ones=2,
                 cumulative=False):
        """Base code taken from:
        https://github.com/anilkagak2/FPTT/blob/main/datasets.py.

        Args:
            N (int, optional): Number of problem instances. Defaults to None, which means that the dataset will be infinite.
            seq_len (int, optional): Single problem instance sequence length. Defaults to 6.
            high (int, optional): Maximum value in the list of inputs. Defaults to 1.
            number_of_ones (int, optional): Number of values to be added. Defaults to 2.
            cumulative (bool, optional): Whether to make expected outcome vector increase by the real input value if the other input is high. Defaults to False.
        """

        self.N = N
        self.seq_len = seq_len
        self.high = high
        self.number_of_ones = number_of_ones
        self.cumulative = cumulative

    def __iter__(self):
        count = 0

        while self.N is None or count < self.N:
            X_num = np.random.uniform(low=0, high=self.high, size=(self.seq_len, 1))
            X_mask = np.zeros((self.seq_len, 1))

            if self.cumulative:
                y_mask = np.zeros((self.seq_len, 1))
            else:
                y_mask = np.zeros((1,))
            
            # Sample positions for 1's in mask
            positions_1 = np.random.choice(np.arange(math.floor(self.seq_len / 2)), 
                                           size=math.floor(self.number_of_ones / 2), 
                                           replace=False)
            positions_2 = np.random.choice(np.arange(math.ceil(self.seq_len / 2), 
                                                     self.seq_len), 
                                           size=math.ceil(self.number_of_ones / 2), 
                                           replace=False)
            
            positions = np.array(list(positions_1) + list(positions_2))
            X_mask[positions] = 1
            
            if self.cumulative:
                local_X_num = np.zeros((self.seq_len, 1))
                local_X_num[positions] = X_num[positions]
                y_mask = np.cumsum(local_X_num, axis=0).flatten()
            else:
                y_mask[0] = np.sum(X_num[positions])
            
            X = np.append(X_num, X_mask, axis=1)

            # Convert X and y_mask to float32 to support MPS devices
            X = X.astype(np.float32)
            y_mask = y_mask.astype(np.float32)

            yield torch.tensor(X.transpose()), torch.tensor(y_mask)

            count += 1

    def show(self, n_examples=2):
        ax = plt.axes()

        colors = []

        examples = list(self.__iter__())

        for i in range(n_examples):
            X, _ = examples[i]

            current_color = next(ax._get_lines.prop_cycler)['color']
            colors.append(current_color)

            plt.plot(list(range(len(X))), X[:, 0], color=current_color)

        # Plot a second time to plot over the noise input values
        for i in range(n_examples):
            X, y = self[i]

            plt.plot(list(range(len(X))),
                     y,
                     color=colors[i],
                     linestyle='dashed')

        plt.xlabel('T')
        plt.ylabel('y')

        plt.show()