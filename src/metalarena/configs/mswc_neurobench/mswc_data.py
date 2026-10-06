from typing import List, Dict, Union, Optional

from autolightning import cc


def create_data_config(
    n_mfcc: int = 32,
    freq: int = 48000, # Hz
    fft_factor: float = 0.05, # n_fft = freq * fft_factor
    win_length: float = 0.032, # seconds
    hop_length: float = 0.016, # seconds
    task_representation: str = 'mels',  # "mfcc" or "mels" "raw"
    resample_rates: Optional[List[float]] = None,
    ms_shift: int = 100, # milliseconds
    apply_spec_augment: bool =  False,
    train_batch_size: int = 256,
    pre_load: bool = False
) -> Dict[str, Union[dict, dict]]:
    
    entries_shift = int(ms_shift / 1000 * freq)
    win_length_int = int(win_length * freq)
    hop_length_int = int(hop_length * freq)
    n_fft_int = int(fft_factor * freq)

    post_batch_tf = None
    spec_augment_tf = None
    take_log = False

    if resample_rates is None:
        resample_rates = []

    if task_representation == "raw":
        batch_transforms = None
    else:
        if task_representation == "mels":
            post_batch_tf = [cc(
                "torchaudio.transforms.MelSpectrogram",
                sample_rate=freq,
                n_mels=n_mfcc,
                win_length=win_length_int,
                hop_length=hop_length_int,
                n_fft=n_fft_int
            )]

            take_log = True
        elif task_representation == "mfcc":
            post_batch_tf = cc(
                "torchaudio.transforms.MFCC",
                sample_rate=freq,
                n_mfcc=n_mfcc,
                log_mels=True,
                melkwargs=dict(
                    win_length=win_length_int,
                    hop_length=hop_length_int,
                    n_fft=n_fft_int
                )
            )
        elif task_representation != "raw":
            raise ValueError(f"Unknown representation: {task_representation}")
        
        batch_transforms = dict(after=post_batch_tf)

    if apply_spec_augment:
        assert task_representation in ["mels", "mfcc"], "SpecAugment is only supported for mels and mfcc"

        spec_augment_tf = cc("torchvision.transforms.Compose", transforms=[
            cc("torchaudio.transforms.TimeMasking", time_mask_param=15),
            cc("torchaudio.transforms.frequencyMasking", freq_mask_param=6)
        ])

        batch_transforms["spec_augment"] = spec_augment_tf

    train_transforms = []

    if resample_rates:
        train_transforms += [cc("torch_mate.data.transforms.RandomResample", orig_freq=freq, resample_rates=resample_rates)]

    if entries_shift != 0:
        train_transforms += [cc("torch_mate.data.transforms.TimeShift", min_shift=-entries_shift, max_shift=entries_shift)]

    config = cc(
        "metalarena.autodatasets.NeurobenchMSWC",
        root="data",
        download=True,
        pre_load=dict(train=True, val=True) if pre_load else False,
        take_log=take_log,
        dataloaders=dict(
            defaults=dict(
                batch_size=train_batch_size,
                num_workers=8,
                prefetch_factor=4,
                persistent_workers=True,
                pin_memory=True,
                drop_last=True
            ),
            train=dict(
                shuffle=True
            ),
            test=dict(
                batch_size=1024,
                drop_last=False
            ),
            val=dict(
                batch_size=1024,
                drop_last=False
            )
        ),
        transforms=dict(
            train=train_transforms
        ),
        batch_transforms=batch_transforms
    )

    return config
