from typing import Union, Literal, Optional

from autolightning import cc


def _create_act_quant_config(collect_stats_steps: int = 1500, float_to_int_impl_type = None, tensor_clamp_impl = None):
    init_args = dict(
        bit_width=4,
        collect_stats_steps=collect_stats_steps,
        # These values are good, but the default ones are also fine
        # Deviating too much does not seem to improve performance
        high_percentile_q = 99.99,
        low_percentile_q = 0.01
    )

    if float_to_int_impl_type is not None:
        init_args["float_to_int_impl_type"] = float_to_int_impl_type

    if tensor_clamp_impl is not None:
        init_args["tensor_clamp_impl"] = tensor_clamp_impl
        

    return cc(
        "autolightning.lm.brevitas.quantizer",
        init_args=dict(
            # ShiftedUint8ActPerTensorFixedPointMSE does not give good results
            # Prepending "BatchQuantStatsScaling1d" unfortunately does not work,
            # there is a bug in Brevitas for this...
            class_paths=["ShiftedUint8ActPerTensorFixedPoint"],
            init_args=init_args
        )
    )


def create_quant_config(
    collect_stats_steps: int = 1500,
    limit_calibration_batches: Optional[int] = None,
    ignore_overflow: Union[bool, Literal["zero_gradients_outside_range"]] = True,
    quant_output: bool = False,
    calibrate: bool = False,
    correct_norms: bool = False,
    correct_biases: bool = False,
    remove_dropout_layers: bool = False,
    skip_linear_layers: bool = False,
    subnet_quant_config: Optional[dict] = None
):
    """This function creates a quantization configuration for quantizing networks
    to run on the Chameleon accelerator (https://github.com/V0XNIHILI/chameleon).

    Args:
        collect_stats_steps (int, optional): _description_. Defaults to 1500.
        limit_calibration_batches (int, optional): _description_. Defaults to None.
        ignore_overflow (Union[bool, Literal["clamp"]], optional): _description_. Defaults to True.
        quant_output (bool, optional): _description_. Defaults to False.
        calibrate (bool, optional): _description_. Defaults to False.
        correct_norms (bool, optional): _description_. Defaults to False.
        correct_biases (bool, optional): _description_. Defaults to False.
        remove_dropout_layers (bool, optional): _description_. Defaults to False.
        skip_linear_layers (bool, optional): _description_. Defaults to False.

    Returns:
        dict: _description_
    """

    assert ignore_overflow in [True, False, "zero_gradients_outside_range"]

    act_quant_config = _create_act_quant_config(
        collect_stats_steps=collect_stats_steps,
        float_to_int_impl_type=cc(
            "brevitas.inject.enum.FloatToIntImplType",
            value="FLOOR"
        ),
        tensor_clamp_impl=cc(
            "brevitas_utils.custom_clamps.ignore_overflow_clamp.IgnoreOverflowClamp",
            gradient_outside_range=0.0 if ignore_overflow == "zero_gradients_outside_range" else 1.0
        ) if ignore_overflow else None
    )

    skip_modules = ["tcn_lib.blocks.Chomp1d", "tcn_lib.blocks.LastElement1d", "tcn_lib.blocks.PointwiseLayer", "tcn_lib.blocks.TemporalBlock", "tcn_lib.blocks.TemporalBottleneck", "tcn_lib.blocks.TemporalConvNet", "tcn_lib.blocks.TemporalLayer", "torch.nn.Dropout", "torch.nn.ZeroPad1d"]

    if skip_linear_layers:
        skip_modules.append("torch.nn.Linear")

    config = dict(
        weight_quant=cc(
            "autolightning.lm.brevitas.quantizer",
            init_args=dict(
                class_paths=["PoT4WeightPerTensorFixedPoint"],
                init_args=dict(
                    bit_width=4,
                    collect_stats_steps=collect_stats_steps
                )
            )
        ),
        act_quant=act_quant_config,
        in_quant=_create_act_quant_config(collect_stats_steps=collect_stats_steps),
        bias_quant=cc(
            "autolightning.lm.brevitas.quantizer",
            init_args=dict(
                class_paths=["Int16Bias"],
                init_args=dict(
                    bit_width=14,
                    collect_stats_steps=collect_stats_steps,
                    narrow_range=False
                )
            )
        ),
        load_float_weights_into_model=True,
        fold_batch_norm_layers=True,
        enable_brevitas_jit=True,
        remove_dropout_layers=remove_dropout_layers,
        allow_quant_tensor_slicing=True,
        skip_modules=skip_modules,
        calibrate=calibrate,
        correct_norms=correct_norms,
        correct_biases=correct_biases,
        limit_calibration_batches=limit_calibration_batches,
        subnet_quant_config=subnet_quant_config
    )

    if quant_output:
        config["out_quant"] = act_quant_config

    return config
