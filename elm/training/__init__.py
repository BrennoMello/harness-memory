"""Self-supervised internal reward estimation and GRPO memory training modules."""

from .internal_reward import (
    extract_majority_vote_consensus,
    compute_internal_rewards,
    RewardResult,
)
from .grpo import GRPOTrainer
from .sft_trainer import SFTMemoryTrainer

__all__ = [
    "extract_majority_vote_consensus",
    "compute_internal_rewards",
    "RewardResult",
    "GRPOTrainer",
    "SFTMemoryTrainer",
]

