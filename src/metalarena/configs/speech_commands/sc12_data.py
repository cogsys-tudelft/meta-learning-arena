from typing import List, Dict, Union, Optional

from autolightning import cc


def create_data_config(
    n_mfcc: int = 32,
    freq: int = 16000, # Hz
    fft_factor: float = 0.05, # n_fft = freq * fft_factor
    win_length: float = 0.032, # seconds
    hop_length: float = 0.016, # seconds
    task_representation: str = 'mels',  # "mfcc" or "mels" "raw"
    normalize: bool = False,
    resample_rates: Optional[List[float]] = None,
    ms_shift: int = 100, # milliseconds
    apply_spec_augment: bool = False,
    noise_p: float = 0.15,
    take_log: bool = True,
    max_noise_level: float = 1.0,
    train_batch_size: int = 256
) -> Dict:
    
    entries_shift = int(ms_shift / 1000 * freq)
    win_length_int = int(win_length * freq)
    hop_length_int = int(hop_length * freq)
    n_fft_int = int(fft_factor * freq)

    post_batch_tf = None
    spec_augment_tf = None

    if resample_rates is None:
        resample_rates = []

    if task_representation == "raw":
        batch_transforms = None

        assert take_log is False, "Log is not supported for raw representation"
    else:
        # Validation loss for mels is lower, but train loss, accuracy and validation
        # accuracy are all the same compared to MFCC
        if task_representation == "mels":
            post_batch_tf = [cc(
                "torchaudio.transforms.MelSpectrogram",
                sample_rate=freq,
                n_mels=n_mfcc,
                win_length=win_length_int,
                hop_length=hop_length_int,
                n_fft=n_fft_int
            )]

            if normalize:
                assert n_mfcc == 32
                assert freq == 16000
                assert fft_factor == 0.05
                assert win_length == 0.032
                assert hop_length == 0.016
                assert resample_rates == []
                assert ms_shift == 100
                assert not apply_spec_augment

                post_batch_tf += [
                    cc(
                        "torch_mate.data.transforms.normalizeSequence",
                        mean=[-4.2381, -3.1812, -2.9908, -3.2680, -3.5574, -3.6632, -3.7244, -3.8747, -4.0679, -4.2519, -4.4514, -4.6733, -4.9138, -5.1001, -5.2508, -5.4285, -5.6380, -5.7985, -5.8381, -5.8370, -5.9325, -5.9988, -6.0202, -6.1127, -6.3004, -6.4734, -6.6417, -6.8070, -6.9329, -7.0094, -7.3057, -8.3728],
                        std=[5.6353, 5.5883, 5.6450, 5.6512, 5.6375, 5.7972, 5.9282, 5.9896, 5.9298, 5.7790, 5.6525, 5.5642, 5.4621, 5.3674, 5.2851, 5.1981, 5.1151, 5.0604, 5.0983, 5.1473, 5.1148, 5.0969, 5.1215, 5.0942, 5.0238, 4.9786, 4.9415, 4.9116, 4.8878, 4.8779, 4.8840, 4.9272]
                    )
                ]
        elif task_representation == "mfcc":
            post_batch_tf = cc(
                "torchaudio.transforms.MFCC",
                sample_rate=freq,
                n_mfcc=n_mfcc,
                log_mels=take_log,
                melkwargs=dict(
                    # n_mels is 128 by default for MelSpectrogram
                    # n_mels=256,
                    win_length=win_length_int,
                    hop_length=hop_length_int,
                    n_fft=n_fft_int
                )
            )

            # Set to false since we do not want to take the log twice in case of MFCC already taking the log
            take_log = False
        elif task_representation != "raw":
            raise ValueError(f"Unknown representation: {task_representation}")
        
        batch_transforms = dict(after=post_batch_tf)

    if normalize and task_representation != 'mels':
        raise ValueError("Normalization is only supported for mels")

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
        "metalarena.autodatasets.SpeechCommandsV2_12C",
        noise_p=noise_p,
        max_noise_level=max_noise_level,
        root="data",
        download=True,
        take_log=take_log,
        dataloaders=dict(
            # There is no shuffling required for the
            # training set as the autodataset already
            # performs balanced sampling
            defaults=dict(
                batch_size=train_batch_size,
                num_workers=8,
                prefetch_factor=4,
                persistent_workers=True,
                pin_memory=True,
                drop_last=True
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
