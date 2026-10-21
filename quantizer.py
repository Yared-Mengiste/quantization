"""Symmetric and asymmetric weight-only INT8 quantization using raw PyTorch.

The notebook contains this same implementation with teaching notes.
Computation is FP32; this is compressed storage, not an INT8 compute kernel.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def quantize_weight(weight, granularity="per_channel", group_size=64, mode="symmetric"):
    """Return INT8 weights, FP32 scales, and optional INT8 zero points."""
    if weight.ndim != 2:
        raise ValueError("Expected [output_features, input_features].")
    if granularity not in {"per_tensor", "per_channel", "per_group"}:
        raise ValueError("Choose per_tensor, per_channel, or per_group.")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer.")
    if mode not in {"symmetric", "asymmetric"}:
        raise ValueError("Choose symmetric or asymmetric.")

    weight = weight.detach().to(torch.float32)
    rows, columns = weight.shape

    # View the data so each scale covers the requested region.
    if granularity == "per_tensor":
        values = weight
        minimum, maximum = values.amin(), values.amax()
    elif granularity == "per_channel":
        values = weight
        minimum = values.amin(dim=1, keepdim=True)  # [rows, 1]
        maximum = values.amax(dim=1, keepdim=True)
    else:
        padding = (-columns) % group_size
        values = F.pad(weight, (0, padding)).reshape(rows, -1, group_size)
        minimum = values.amin(dim=2, keepdim=True)  # [rows, groups, 1]
        maximum = values.amax(dim=2, keepdim=True)

    if mode == "symmetric":
        max_abs = torch.maximum(minimum.abs(), maximum.abs())
        scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
        zero_point = None  # Zero is implicit; no metadata needed.
        integers = torch.round(values / scale).clamp(-127, 127).to(torch.int8)
    else:
        q_min, q_max = -128, 127
        # Include real zero BEFORE finding the scale. Just clipping a zero point
        # with the old scale could clip positive-only or negative-only weights.
        minimum = torch.minimum(minimum, torch.zeros_like(minimum))
        maximum = torch.maximum(maximum, torch.zeros_like(maximum))
        spread = maximum - minimum
        scale = torch.where(spread > 0, spread / (q_max - q_min), torch.ones_like(spread))
        raw_zero_point = torch.round(q_min - minimum / scale)
        # Cap above 127 to 127; cap below -128 to -128, before converting dtype.
        zero_point = raw_zero_point.clamp(q_min, q_max).to(torch.int8)
        integers = (torch.round(values / scale) + zero_point.float()).clamp(q_min, q_max).to(torch.int8)

    if granularity == "per_group":
        # Padding was only needed to compute scales; do not store fake weights.
        integers = integers.reshape(rows, -1)[:, :columns].contiguous()

    return integers, scale, zero_point


def dequantize_weight(integers, scale, granularity="per_channel", group_size=64, zero_point=None):
    """Reconstruct an FP32 matrix, broadcasting each scale over its region."""
    values = integers.to(torch.float32)
    if granularity != "per_group":
        if zero_point is not None:
            values = values - zero_point.float()
        return values * scale

    rows, columns = values.shape
    padding = (-columns) % group_size
    groups = F.pad(values, (0, padding)).reshape(rows, -1, group_size)
    if zero_point is not None:
        groups = groups - zero_point.float()
    return (groups * scale).reshape(rows, -1)[:, :columns].contiguous()


class Int8Linear(nn.Module):
    """Stores INT8 weights; reconstructs them for ordinary FP32 inference."""

    def __init__(self, linear, granularity="per_channel", group_size=64, mode="symmetric"):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.granularity = granularity
        self.group_size = group_size
        self.mode = mode
        integers, scale, zero_point = quantize_weight(linear.weight, granularity, group_size, mode)
        # Buffers are saved in state_dict but are not trainable parameters.
        self.register_buffer("qweight", integers)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        bias = None if linear.bias is None else linear.bias.detach().float().clone()
        self.register_buffer("bias", bias)

    def reconstructed_weight(self):
        return dequantize_weight(
            self.qweight, self.scale, self.granularity, self.group_size, self.zero_point
        )

    def forward(self, inputs):
        # Do not cache a permanent FP32 copy: that would undo memory savings.
        return F.linear(inputs, self.reconstructed_weight(), self.bias)


def quantize_linears(model, granularity="per_channel", group_size=64, mode="symmetric"):
    """Replace linear layers in-place, preserving the final classifier.

    The caller should copy the FP32 model first when comparing configurations.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name != "classifier":
            setattr(model, name, Int8Linear(child, granularity, group_size, mode))
        else:
            quantize_linears(child, granularity, group_size, mode)
    return model


def tensor_storage_bytes(model):
    """Tensor payload only; excludes Python objects and temporary activations."""
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def weight_rmse(original, quantized):
    """Global RMSE over only the linear weights changed by our quantizer."""
    originals = dict(original.named_modules())
    squared_error, count = 0.0, 0
    for name, layer in quantized.named_modules():
        if isinstance(layer, Int8Linear):
            weight = originals[name].weight.detach()
            difference = weight - layer.reconstructed_weight()
            squared_error += difference.square().sum().item()
            count += weight.numel()
    return (squared_error / count) ** 0.5 if count else 0.0


def save_quantized(model, path, granularity, group_size=64, mode="symmetric"):
    """Save only state tensors and the small amount of loader metadata."""
    torch.save(
        {
            "format_version": 2,
            "mode": mode,
            "granularity": granularity,
            "group_size": group_size,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_quantized(folder):
    """Load an exported local model folder; no pretrained FP32 download."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    folder = Path(folder)
    checkpoint = torch.load(folder / "quantized.pt", map_location="cpu", weights_only=True)
    if checkpoint["format_version"] not in {1, 2}:
        raise ValueError("Unsupported checkpoint format.")

    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    # Construct the architecture, install our layer types, then fill their buffers.
    # This simple loader briefly allocates randomly initialized FP32 layers.
    model = AutoModelForSequenceClassification.from_config(
        config, attn_implementation="eager"
    ).float().cpu()
    quantize_linears(
        model, checkpoint["granularity"], checkpoint["group_size"],
        mode=checkpoint.get("mode", "symmetric"),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()
