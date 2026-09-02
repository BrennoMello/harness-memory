"""Memory Key Indexing and Retrieval mechanisms (ELM_input and ELM_kv).

Paper specifications:
- Input Retrieval: query q = mean_t(E(x)) in R^D, key k^(i) = mean_t(m^(i)).
  Score: sim(q, k^(i)) via cosine similarity.
- KV Retrieval: query q^(l) = mean_{heads, seq}(QKV(x)^(l)) for each layer l.
  Score: 1/L * sum_{l=1}^L sim(q^(l), k^(i,l)).
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from .pool import MemoryPool, MemoryEntry


class MemoryIndex:
    """Index for querying memories in MemoryPool via Input or KV representation."""
    
    def __init__(self, retrieval_mode: str = "input"):
        self.retrieval_mode = retrieval_mode.lower()
        
    def retrieve(
        self,
        memory_pool: MemoryPool,
        query_input_key: torch.Tensor,
        query_kv_keys: Optional[List[torch.Tensor]] = None,
        top_k: int = 1,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve the top-k most similar memories from pool M."""
        if len(memory_pool) == 0:
            return []
            
        entries = memory_pool.get_all_entries()
        
        if self.retrieval_mode == "input" or query_kv_keys is None:
            # Input key similarity: cosine(q, k^(i))
            q = F.normalize(query_input_key.float().unsqueeze(0), p=2, dim=-1)  # (1, D)
            
            keys = torch.stack([e.input_key.float() for e in entries], dim=0)    # (N, D)
            keys = F.normalize(keys, p=2, dim=-1)
            
            sims = torch.mm(q, keys.transpose(0, 1)).squeeze(0)  # (N,)
            top_vals, top_indices = torch.topk(sims, k=min(top_k, len(entries)))
            
            results = []
            for val, idx in zip(top_vals, top_indices):
                results.append((entries[idx.item()], val.item()))
            return results
            
        elif self.retrieval_mode == "kv":
            # Layer-wise KV similarity: 1/L * sum_{l=1}^L cosine(q^(l), k^(i,l))
            num_layers = len(query_kv_keys)
            scores = torch.zeros(len(entries), dtype=torch.float32, device=query_input_key.device)
            
            for l in range(num_layers):
                ql = F.normalize(query_kv_keys[l].float().unsqueeze(0), p=2, dim=-1)  # (1, D)
                layer_keys = torch.stack([
                    e.kv_keys[l].float() if (e.kv_keys and len(e.kv_keys) > l) else e.input_key.float()
                    for e in entries
                ], dim=0)  # (N, D)
                layer_keys = F.normalize(layer_keys, p=2, dim=-1)
                
                layer_sim = torch.mm(ql, layer_keys.transpose(0, 1)).squeeze(0)
                scores += layer_sim
                
            scores = scores / num_layers
            top_vals, top_indices = torch.topk(scores, k=min(top_k, len(entries)))
            
            results = []
            for val, idx in zip(top_vals, top_indices):
                results.append((entries[idx.item()], val.item()))
            return results
            
        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")
            
    @staticmethod
    def extract_input_query(model: torch.nn.Module, tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
        """Extract mean-pooled input embedding vector q in R^D."""
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt").to(device)
            # Embedding layer
            if hasattr(model, "get_input_embeddings"):
                embed_layer = model.get_input_embeddings()
                embeds = embed_layer(inputs["input_ids"])  # (1, seq_len, D)
            elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                embeds = model.model.embed_tokens(inputs["input_ids"])
            else:
                raise AttributeError("Could not locate embedding layer on model")
                
            # Mean pool over sequence dimension: meant(E(x))
            q = torch.mean(embeds[0], dim=0).detach()
            return q
            
    @staticmethod
    def extract_kv_query(model: torch.nn.Module, tokenizer: Any, text: str, device: torch.device) -> List[torch.Tensor]:
        """Extract layer-wise pooled QKV activations for KV retrieval."""
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            # Use hidden states across layers as representation
            hidden_states = outputs.hidden_states[1:]  # Skip input embedding layer
            
            layer_keys = []
            for h in hidden_states:
                # h: (1, seq_len, D) -> mean pool over seq_len -> (D,)
                layer_keys.append(torch.mean(h[0], dim=0).detach())
            return layer_keys
