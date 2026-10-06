import torch
from torch.utils.data import IterableDataset
from typing import Iterator


class CosineClosest(IterableDataset):
    def __init__(self, n: int, feature_dim: int):
        self.n = n
        self.feature_dim = feature_dim

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        while True:
            # Generate n + 1 feature vectors from a normal distribution
            features = torch.randn(self.n + 1, self.feature_dim)
            # Calculate cosine similarity between (n+1)th vector and first n vectors
            cosine_similarities = torch.nn.functional.cosine_similarity(
                features[:-1], features[-1].unsqueeze(0), dim=1
            )
            # Label: index of the feature vector closest to the (n+1)th
            label = cosine_similarities.argmax().item()
            yield features.flatten(), label
