"""Group Relative Policy Optimization (GRPO) implementation for ELM memory training.

Implements Eq. (1) advantage normalization and DR-GRPO surrogate loss with KL regularization.
Trains only the lightweight modular memory parameters (soft prompt / LoRA) while base model is frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from ..config import GRPOConfig
from ..memory.soft_prompt import SoftPromptMemory


class GRPOTrainer:
    """GRPO policy optimization trainer for individual sample memory distillation."""
    
    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        learning_rate: float = 1e-4,
        device: torch.device = torch.device("cpu"),
    ):
        self.config = config or GRPOConfig()
        self.learning_rate = learning_rate
        self.device = device
        
    def compute_group_advantages(self, rewards: List[float]) -> torch.Tensor:
        """Compute normalized advantage values for rollouts in a group (Eq. 1).
        
        A_hat_i = (r_i - mean(r)) / (std(r) + eps)
        """
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        mean_r = torch.mean(r)
        std_r = torch.std(r, unbiased=False)
        
        # If all rewards are identical (std == 0), advantages are zero
        if std_r < 1e-8:
            return torch.zeros_like(r)
            
        advantages = (r - mean_r) / (std_r + 1e-8)
        return advantages

    def train_soft_prompt_memory(
        self,
        memory: SoftPromptMemory,
        model: nn.Module,
        tokenizer: Any,
        prompt_text: str,
        rollout_texts: List[str],
        rewards: List[float],
        train_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train soft prompt memory for S gradient steps using GRPO on candidate rollouts.
        
        Args:
            memory: SoftPromptMemory module to optimize.
            model: Frozen base language model.
            tokenizer: Tokenizer for formatting and token IDs.
            prompt_text: Input question prompt.
            rollout_texts: Sampled response completions (K=8).
            rewards: Binary rewards for each completion.
            train_steps: Number of update steps (default 10).
        """
        steps = train_steps or self.config.train_steps
        optimizer = torch.optim.AdamW(
            [memory.soft_tokens],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        advantages = self.compute_group_advantages(rewards)
        
        # Tokenize prompt and rollouts
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids.size(1)
        
        # Prepare batch of sequences
        batch_input_ids = []
        batch_labels = []
        
        for rollout in rollout_texts:
            full_text = prompt_text + rollout
            full_ids = tokenizer.encode(full_text, return_tensors="pt").to(self.device)
            
            # Truncate if exceeding max_length
            if full_ids.size(1) > self.config.max_length:
                full_ids = full_ids[:, : self.config.max_length]
                
            labels = full_ids.clone()
            # Mask out prompt tokens so loss is only computed on generated response
            labels[:, :prompt_len] = -100
            
            batch_input_ids.append(full_ids)
            batch_labels.append(labels)
            
        # Pad sequences to uniform length in batch
        max_seq_len = max(x.size(1) for x in batch_input_ids)
        pad_id = tokenizer.pad_token_id or 0
        
        padded_ids = torch.full((len(rollout_texts), max_seq_len), pad_id, dtype=torch.long, device=self.device)
        padded_labels = torch.full((len(rollout_texts), max_seq_len), -100, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(rollout_texts), max_seq_len), dtype=torch.long, device=self.device)
        
        for i, (ids, labs) in enumerate(zip(batch_input_ids, batch_labels)):
            cur_len = ids.size(1)
            padded_ids[i, :cur_len] = ids[0]
            padded_labels[i, :cur_len] = labs[0]
            attention_mask[i, :cur_len] = 1
            
        # Compute reference (base model without memory or frozen state) log probabilities
        with torch.no_grad():
            ref_outputs = model(input_ids=padded_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits[:, :-1, :]  # Shift for next-token prediction
            ref_targets = padded_labels[:, 1:]          # Shifted targets
            
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            # Gather log probs for target tokens
            target_mask = (ref_targets != -100)
            safe_targets = ref_targets.clamp(min=0)
            ref_token_log_probs = torch.gather(ref_log_probs, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
            ref_token_log_probs = ref_token_log_probs * target_mask.float()
            ref_seq_log_probs = ref_token_log_probs.sum(dim=-1) / (target_mask.float().sum(dim=-1) + 1e-8)

        # Get base model embedding layer
        if hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_layer = model.model.embed_tokens
        else:
            embed_layer = None

        loss_history = []
        
        # Optimization Loop for S steps
        for step in range(steps):
            optimizer.zero_grad()
            
            if embed_layer is not None:
                token_embeds = embed_layer(padded_ids)  # (batch, seq_len, D)
                extended_embeds, extended_mask = memory.forward_embeddings(token_embeds, attention_mask)
                outputs = model(inputs_embeds=extended_embeds, attention_mask=extended_mask)
                
                # Output logits are for [k_tokens + seq_len], slice to seq_len
                logits = outputs.logits[:, memory.num_tokens - 1 : -1, :]
            else:
                # Fallback for dummy/mock model
                outputs = model(input_ids=padded_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :]
                
            cur_log_probs = F.log_softmax(logits, dim=-1)
            cur_token_log_probs = torch.gather(cur_log_probs, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
            cur_token_log_probs = cur_token_log_probs * target_mask.float()
            cur_seq_log_probs = cur_token_log_probs.sum(dim=-1) / (target_mask.float().sum(dim=-1) + 1e-8)
            
            # Policy ratio: exp(log_pi - log_pi_ref)
            ratio = torch.exp(cur_seq_log_probs - ref_seq_log_probs.detach())
            
            # Clipped surrogate GRPO objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL divergence penalty: D_KL(pi || pi_ref) approx (ratio - 1) - log(ratio)
            kl_div = torch.mean(ratio - 1.0 - (cur_seq_log_probs - ref_seq_log_probs.detach()))
            
            total_loss = policy_loss + self.config.kl_penalty_beta * kl_div
            
            total_loss.backward()
            optimizer.step()
            
            loss_history.append({
                "step": step,
                "total_loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "kl_div": kl_div.item(),
            })
            
        return {
            "steps": steps,
            "final_loss": loss_history[-1]["total_loss"] if loss_history else 0.0,
            "loss_history": loss_history,
        }

