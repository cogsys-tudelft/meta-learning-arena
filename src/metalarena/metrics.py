import torch
import torchmetrics


class BinaryRecallSpecificityAverage(torchmetrics.Metric):
    """Metric that computes the average of recall (sensitivity) and specificity for binary classification.
    Useful for sweeps where we want a single metric that balances both classes, especially in imbalanced datasets.
    """
    def __init__(self) -> None:
        super().__init__()

        self.specificity = torchmetrics.Specificity(
            task="multiclass",
            num_classes=2,
            ignore_index=1,  # negative class accuracy
        )

        self.sensitivity = torchmetrics.Recall(
            task="multiclass",
            num_classes=2,
            ignore_index=0,  # positive class accuracy
        )

    def update(
        self,
        logits: torch.Tensor,  # (batch, 2)
        targets: torch.Tensor  # (batch,)
    ) -> None:
        # --- Input validation ---
        assert logits.ndim == 2, f"logits must be 2D, got {logits.shape}"
        assert logits.size(1) == 2, f"logits second dim must be 2, got {logits.size(1)}"
        assert targets.ndim == 1, f"targets must be 1D, got {targets.shape}"
        assert logits.size(0) == targets.size(0), "batch size mismatch"

        # Update internal metrics
        self.specificity.update(logits, targets)
        self.sensitivity.update(logits, targets)

    def compute(self) -> torch.Tensor:
        spec = self.specificity.compute()
        sens = self.sensitivity.compute()
        return 0.5 * (spec + sens)

    def reset(self) -> None:
        self.specificity.reset()
        self.sensitivity.reset()
        super().reset()
