from pathlib import Path
from typing import Any, Dict

import torch

_WEIGHT_PATH = Path(__file__).resolve().parents[3] / "predictor" / "weight.txt"


def FFT(x, cutoff_ratio=0.1):
    orig_dtype = x.dtype
    device = x.device
    x_fp32 = x.to(torch.float32)

    B, HW, D = x_fp32.shape
    freq = torch.fft.fft(x_fp32, dim=1)
    freqs = torch.fft.fftfreq(HW, d=1.0, device=device)
    cutoff = cutoff_ratio * freqs.abs().max()

    low_mask = (freqs.abs() <= cutoff)[None, :, None]
    high_mask = ~low_mask

    low_fp32 = torch.fft.ifft(freq * low_mask, dim=1).real
    high_fp32 = torch.fft.ifft(freq * high_mask, dim=1).real

    low = low_fp32.to(device=device, dtype=orig_dtype)
    high = high_fp32.to(device=device, dtype=orig_dtype)
    return low, high


def init_learnable(h: torch.Tensor, l: torch.Tensor, x: torch.Tensor) -> Dict[str, Any]:
    return {"h": [h], "l": [l], "x": [x.to(torch.float32).reshape(x.shape[0], 1, -1)]}


def update_learnable(cache_dic: Dict[str, Any], current: Dict[str, Any], x: torch.Tensor) -> None:
    module = current["module"]
    key = f"{module}_learnable"
    l, h = FFT(x.to(torch.float32))
    h = h.reshape(h.shape[0], 1, -1)
    l = l.reshape(l.shape[0], 1, -1)
    if key not in cache_dic:
        cache_dic[key] = init_learnable(h, l, x)
        return
    cache_dic[key]["h"].append(h)
    cache_dic[key]["l"].append(l)
    cache_dic[key]["x"].append(x.to(torch.float32).reshape(x.shape[0], 1, -1))
    cache_dic[key]["type"] = x.dtype
    cache_dic[key]["shape"] = x.shape


def predict_with_learnable(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> torch.Tensor:
    module = current["module"]
    key = f"{module}_learnable"
    if key not in cache_dic:
        raise KeyError(f"{key} not found. Call update_learnable first.")

    x = cache_dic[key]["x"]
    x = torch.cat(x, dim=1)
    T = x.shape[1]

    with open(_WEIGHT_PATH, "r") as f:
        weight = torch.tensor(eval(f.read()), dtype=torch.float32, device=x.device)[T - 1][:T]
    pred = torch.matmul(weight, x)
    return pred.reshape(cache_dic[key]["shape"]).to(cache_dic[key]["type"])
