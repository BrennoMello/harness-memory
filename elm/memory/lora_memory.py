"""LoRA Memory parameterization (Ablation).

Uses PEFT LoRA adapters (r=16) as per-sample memory units.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from ..config import LoRAConfig


class LoRAMemory:
    """LoRA adapter per-sample memory unit."""
    
    def __init__(
        self,
        config: Optional[LoRAConfig] = None,
        adapter_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.config = config or LoRAConfig()
        self.adapter_state_dict = adapter_state_dict or {}
        self.stored_key: Optional[torch.Tensor] = None
        
    def save_adapter_weights(self, peft_model: Any, key_tensor: Optional[torch.Tensor] = None):
        """Extract and store trainable adapter weights and key representation."""
        self.adapter_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in peft_model.named_parameters()
            if "lora" in k.lower() and v.requires_grad
        }
        if key_tensor is not None:
            self.stored_key = key_tensor.detach().cpu().clone()

    def load_into_model(self, peft_model: Any):
        """Load stored adapter weights into active PEFT model."""
        state_dict = {k: v.to(peft_model.device) for k, v in self.adapter_state_dict.items()}
        peft_model.load_state_dict(state_dict, strict=False)

    def get_input_key(self) -> Optional[torch.Tensor]:
        return self.stored_key

