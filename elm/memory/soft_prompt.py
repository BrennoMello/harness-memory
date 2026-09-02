"""Soft Prompt Memory implementation for ELM.

Parametrizes memory as a sequence of k=20 trainable continuous vectors
prepended to input embeddings, keeping the base model completely frozen.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SoftPromptMemory(nn.Module):
    """Modular Soft Prompt Memory unit (m_x in R^{k x d})."""
    
    def __init__(
        self,
        num_tokens: int = 20,
        hidden_dim: int = 4096,
        init_embeddings: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # Trainable soft prompt tokens: (k, d)
        self.soft_tokens = nn.Parameter(
            torch.empty((num_tokens, hidden_dim), dtype=dtype, device=device)
        )
        
        # Initialize
        if init_embeddings is not None:
            self.init_from_embeddings(init_embeddings)
        else:
            # Random initialization with small std
            nn.init.normal_(self.soft_tokens, mean=0.0, std=0.02)
            
    def init_from_embeddings(self, embeddings: torch.Tensor):
        """Initialize soft prompt tokens from input question embeddings."""
        with torch.no_grad():
            # embeddings shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
            if embeddings.dim() == 3:
                embeddings = embeddings[0]  # Take first sequence
            
            seq_len = embeddings.size(0)
            if seq_len >= self.num_tokens:
                # Take first num_tokens
                self.soft_tokens.data.copy_(embeddings[: self.num_tokens].to(self.soft_tokens.dtype))
            else:
                # Repeat or tile if sequence is shorter than num_tokens
                repeats = (self.num_tokens + seq_len - 1) // seq_len
                tiled = embeddings.repeat(repeats, 1)[: self.num_tokens]
                self.soft_tokens.data.copy_(tiled.to(self.soft_tokens.dtype))

    def forward_embeddings(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepend soft prompt tokens to the input embeddings.
        
        Args:
            input_embeds: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            extended_embeds: (batch_size, num_tokens + seq_len, hidden_dim)
            extended_attention_mask: (batch_size, num_tokens + seq_len)
        """
        batch_size = input_embeds.size(0)
        
        # Expand soft tokens to batch dimension: (batch_size, num_tokens, hidden_dim)
        expanded_prompt = self.soft_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate along sequence dimension: [p_1, ..., p_k, x]
        extended_embeds = torch.cat([expanded_prompt, input_embeds], dim=1)
        
        extended_mask = None
        if attention_mask is not None:
            prompt_mask = torch.ones(
                (batch_size, self.num_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            
        return extended_embeds, extended_mask

    def get_input_key(self) -> torch.Tensor:
        """Derive memory key for Input Retrieval by mean pooling over soft prompt tokens."""
        # Key vector in R^d
        with torch.no_grad():
            return torch.mean(self.soft_tokens, dim=0).detach().clone()

    def clone_memory(self) -> "SoftPromptMemory":
        """Create an exact detached copy of this memory unit."""
        new_mem = SoftPromptMemory(
            num_tokens=self.num_tokens,
            hidden_dim=self.hidden_dim,
            device=self.soft_tokens.device,
            dtype=self.soft_tokens.dtype,
        )
        with torch.no_grad():
            new_mem.soft_tokens.data.copy_(self.soft_tokens.data)
        return new_mem

