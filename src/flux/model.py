from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flux.learnable_utils import predict_with_learnable, update_learnable
from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from flux.modules.lora import LinearLora, replace_linear_with_lora

# Full-compute steps for each cache interval (50-step schedule).
FULL_STEP_SCHEDULES = {
    1: list(range(50)),
    5: [0, 1, 2, 4, 6, 9, 14, 23, 30, 37, 44],
    6: [0, 1, 2, 4, 7, 14, 23, 31, 38, 45],
    7: [0, 1, 2, 4, 9, 17, 26, 35, 43],
    8: [0, 1, 2, 4, 10, 20, 31, 41],
    10: [0, 1, 2, 4, 12, 24, 38],
    16: [0, 1, 2, 4, 16, 32],
}


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """FLUX transformer with L2P output-level caching."""

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        *args,
        **kwargs,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        cache_dic = kwargs.get("cache_dic", None)
        current = kwargs.get("current", None)

        if cache_dic is None:
            return self._forward_full(img, img_ids, txt, txt_ids, timesteps, y, guidance)

        interval = cache_dic["fresh_threshold"]
        if interval not in FULL_STEP_SCHEDULES:
            raise ValueError(f"Unsupported cache interval: {interval}")

        full_steps = FULL_STEP_SCHEDULES[interval]
        current["type"] = "full" if current["step"] in full_steps else "cache"

        if current["type"] == "full":
            img = self._forward_full(img, img_ids, txt, txt_ids, timesteps, y, guidance, cache_dic, current)
            update_learnable(cache_dic=cache_dic, current=current, x=img)
        else:
            img = predict_with_learnable(cache_dic=cache_dic, current=current)
            update_learnable(cache_dic=cache_dic, current=current, x=img)

        return img

    def _forward_full(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None,
        cache_dic=None,
        current=None,
    ) -> Tensor:
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        block_kwargs = {}
        if cache_dic is not None:
            block_kwargs = {"cache_dic": cache_dic, "current": current}

        for i, block in enumerate(self.double_blocks):
            if current is not None:
                current["layer"] = i
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, **block_kwargs)

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if current is not None:
                current["layer"] = i
            img = block(img, vec=vec, pe=pe, **block_kwargs)

        img = img[:, txt.shape[1] :, ...]
        return self.final_layer(img, vec)


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
