"""Supervised Fine-Tuning (SFT) memory training baseline (Ablation).

Trains soft prompt on greedy response or majority vote response using standard cross-entropy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from ..memory.soft_prompt import SoftPromptMemory


class SFTMemoryTrainer:
    """SFT trainer for soft prompt memory ablation."""
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        device: torch.device = torch.device("cpu"),
    ):
        self.learning_rate = learning_rate
        self.device = device
        
    def train_soft_prompt(
        self,
        memory: SoftPromptMemory,
        model: nn.Module,
        tokenizer: Any,
        prompt_text: str,
        target_solution: str,
        train_steps: int = 10,
    ) -> Dict[str, Any]:
        """Train soft prompt using SFT cross-entropy loss on target solution."""
        optimizer = torch.optim.AdamW([memory.soft_tokens], lr=self.learning_rate)
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids.size(1)
        
        full_text = prompt_text + target_solution
        full_ids = tokenizer.encode(full_text, return_tensors="pt").to(self.device)
        
        labels = full_ids.clone()
        labels[:, :prompt_len] = -100  # Mask prompt
        
        if hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_layer = model.model.embed_tokens
        else:
            embed_layer = None
            
        loss_history = []
        for step in range(train_steps):
            optimizer.zero_grad()
            
            if embed_layer is not None:
                token_embeds = embed_layer(full_ids)
                extended_embeds, _ = memory.forward_embeddings(token_embeds)
                outputs = model(inputs_embeds=extended_embeds)
                logits = outputs.logits[:, memory.num_tokens - 1 : -1, :]
            else:
                outputs = model(input_ids=full_ids)
                logits = outputs.logits[:, :-1, :]
                
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
        return {"steps": train_steps, "final_loss": loss_history[-1] if loss_history else 0.0}

