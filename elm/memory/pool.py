"""Memory Pool management for ELM.

Stores, saves, loads, and manages the collection of learned latent memories M = {m^(i)}.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from .soft_prompt import SoftPromptMemory
from .lora_memory import LoRAMemory
from .prefix_tuning import PrefixTuningMemory


@dataclass
class MemoryEntry:
    """A single memory entry stored in the memory pool."""
    memory_id: str
    source_sample_id: str
    source_question: str
    consensus_answer: Optional[str]
    input_key: torch.Tensor                     # Key vector in R^D (ELM_input)
    kv_keys: Optional[List[torch.Tensor]] = None # Layer-wise keys in R^D (ELM_kv)
    memory_module: Optional[Union[SoftPromptMemory, LoRAMemory, PrefixTuningMemory]] = None
    created_at_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryPool:
    """The Memory Pool M that accumulates modular memories across test-time stream."""
    
    def __init__(self):
        self.entries: List[MemoryEntry] = []
        self._id_map: Dict[str, MemoryEntry] = {}
        
    def __len__(self) -> int:
        return len(self.entries)
        
    def add(self, entry: MemoryEntry):
        """Add a newly created memory to the pool."""
        self.entries.append(entry)
        self._id_map[entry.memory_id] = entry
        
    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        return self._id_map.get(memory_id)
        
    def get_all_entries(self) -> List[MemoryEntry]:
        return self.entries
        
    def clear(self):
        """Empty the memory pool."""
        self.entries.clear()
        self._id_map.clear()
        
    def save_to_disk(self, save_path: str):
        """Persist memory pool state and tensors to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        serializable_data = []
        for entry in self.entries:
            module_state = None
            if isinstance(entry.memory_module, SoftPromptMemory):
                module_state = {
                    "type": "soft_prompt",
                    "num_tokens": entry.memory_module.num_tokens,
                    "hidden_dim": entry.memory_module.hidden_dim,
                    "tokens": entry.memory_module.soft_tokens.data.cpu(),
                }
            elif isinstance(entry.memory_module, LoRAMemory):
                module_state = {
                    "type": "lora",
                    "state_dict": entry.memory_module.adapter_state_dict,
                }
                
            serializable_data.append({
                "memory_id": entry.memory_id,
                "source_sample_id": entry.source_sample_id,
                "source_question": entry.source_question,
                "consensus_answer": entry.consensus_answer,
                "input_key": entry.input_key.cpu(),
                "kv_keys": [k.cpu() for k in entry.kv_keys] if entry.kv_keys else None,
                "module_state": module_state,
                "metadata": entry.metadata,
            })
        torch.save(serializable_data, save_path)

    def load_from_disk(self, load_path: str, device: torch.device = torch.device("cpu")):
        """Load memory pool state from disk."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Memory pool file not found: {load_path}")
            
        data = torch.load(load_path, map_location=device)
        self.clear()
        for item in data:
            module = None
            m_state = item.get("module_state")
            if m_state and m_state.get("type") == "soft_prompt":
                module = SoftPromptMemory(
                    num_tokens=m_state["num_tokens"],
                    hidden_dim=m_state["hidden_dim"],
                    device=device,
                )
                module.soft_tokens.data.copy_(m_state["tokens"].to(device))
            elif m_state and m_state.get("type") == "lora":
                module = LoRAMemory(adapter_state_dict=m_state["state_dict"])
                
            entry = MemoryEntry(
                memory_id=item["memory_id"],
                source_sample_id=item["source_sample_id"],
                source_question=item["source_question"],
                consensus_answer=item["consensus_answer"],
                input_key=item["input_key"].to(device),
                kv_keys=[k.to(device) for k in item["kv_keys"]] if item.get("kv_keys") else None,
                memory_module=module,
                metadata=item.get("metadata", {}),
            )
            self.add(entry)

