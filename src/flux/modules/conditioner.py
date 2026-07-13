import json
import os

from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


def _is_clip_model(model_path: str) -> bool:
    if "openai" in model_path or "clip" in model_path.lower():
        return True
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            architectures = json.load(f).get("architectures", [])
        return any("CLIP" in arch for arch in architectures)
    return False


class HFEmbedder(nn.Module):
    def __init__(
        self,
        version: str,
        max_length: int,
        tokenizer_path: str | None = None,
        **hf_kwargs,
    ):
        super().__init__()
        self.is_clip = _is_clip_model(version)
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        tok_path = tokenizer_path or version

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tok_path, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(tok_path, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
