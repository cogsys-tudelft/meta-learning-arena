from autolightning.dm.few_shot import FewShotMixin

from metalarena.autodatasets.lightning_audio_dataset import LightningAudioDataset


class NeurobenchMSWC(LightningAudioDataset):
   def __init__(self,
                 root: str,
                 download: bool = False,
                 subset: str = 'base', # 'base' or 'evaluation'
                 **kwargs):
        super().__init__(
            dataset={
                "class_name": "neurobench.datasets.MSWC_dataset.MSWC",
                "args": dict(
                    defaults=dict(
                        root=root,
                        subset=subset,
                        download=download,
                        procedure='training'
                    ),
                    # The procedure argument is ignored for the 'evaluation' subset,
                    # so we can still specify it even when subset='evaluation'.
                    train=dict(procedure='training'),
                    val=dict(procedure='validation'),
                    test=dict(procedure='testing')
                )
            },
            random_split=None,
            requires_prepare=download,
            target_batch_transforms=None,
            **kwargs
        )
