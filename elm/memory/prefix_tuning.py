"""Prefix Tuning Memory parameterization (Ablation)."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class PrefixTuningMemory(nn.Module):
    """Prefix tuning memory unit attached to KV layers."""
    
    def __init__(
        self,
        num_prefix_tokens: int = 20,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # (num_layers, 2, num_heads, num_prefix_tokens, head_dim) for K and V
        self.prefix_kv = nn.Parameter(
            torch.empty(
                (num_layers, 2, num_heads, num_prefix_tokens, head_dim),
                dtype=dtype,
                device=device,
            )
        )
        nn.init.normal_(self.prefix_kv, mean=0.0, std=0.02)
        
    def get_input_key(self) -> torch.Tensor:
        """Mean pool across all layers and heads to produce key."""
        with torch.no_grad():
            return torch.mean(self.prefix_kv, dim=(0, 1, 2, 3)).detach().clone()

