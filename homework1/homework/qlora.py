from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit

# Copilot was uesd to guide.
class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)
        
        # Initialize weights
        torch.nn.init.kaiming_uniform_(self.lora_a.weight.data)
        torch.nn.init.zeros_(self.lora_b.weight.data)

        # Scaling factor hyperparameter for LoRA influence
        # 8 weak, 16 strong (popular), 32 very strong
        alpha = 32
        self.alpha_div_rank = alpha / lora_dim

        # Make linear layers non-trainable but ensure LoRA parameters remain trainable
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
        x_dtype = x.dtype

        # Ensure variables are on the same device
        device = x.device
        if self.lora_a.weight.device != device:
            self.lora_a.to(device)
            self.lora_b.to(device)

        x_linear_dtype = x.to(device, dtype=self.linear_dtype)

        result = super().forward(x_linear_dtype).to(device,  dtype=x_dtype) + self.alpha_div_rank * self.lora_b(self.lora_a(x))
        return result

class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size, bias=True),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size, bias=True),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size, bias=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 8, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
