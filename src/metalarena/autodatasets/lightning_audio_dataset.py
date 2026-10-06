import torch

from autolightning import AutoDataModule


class LightningAudioDataset(AutoDataModule):
    def __init__(self, take_log: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.take_log = take_log

        # Add support for taking a possible log of the melspectogram
        if self.take_log:
            self.possible_log_forward = lambda x : torch.log(x + 1e-9) 
        else:
             self.possible_log_forward = lambda x : x
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        # Make sure post batch transfer transform only get applied to the X data
        # and if an Melspec/MFCC transform has been applied that the data gets
        # reshaped correctly

        x, y = batch

        if "after" in self.batch_transforms:
            bt_after = self.batch_transforms["after"]

            # If the batch transform is a list, we only want the first element
            # as this is the MFCC/Melspec transform
            if type(bt_after) == list:
                bt_after = bt_after[0]

            if hasattr(bt_after, 'to'):
                bt_after.to(x.device)

        if x.shape[-1] == 1:
            x = torch.transpose(x, -1, -2)

        x, y = super().on_after_batch_transfer((x, y), dataloader_idx)

        spec_augment = self.batch_transforms.get("spec_augment", None)

        # Only apply spec augment if we are in training mode
        if spec_augment is not None and torch.is_grad_enabled():
            x = spec_augment(x)

        if len(x.shape) == 4:
            # Shape of x right now is: (batch_size, 1, n_mffc, time)
            batch_size, _, channels, time = x.shape

            # But we want it to be (batch_size, time) otherwise the output
            # is not supported by the nn.Conv1D operation.
            x = x.reshape(batch_size, channels, time)

        x = self.possible_log_forward(x)

        return x, y
