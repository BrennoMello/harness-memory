"""Configuration dataclasses for ELM replication."""

from dataclasses import dataclass, field
from typing import Literal, Optional, List


@dataclass
class PromptConfig:
    """Prompt formatting and system prompt configuration."""
    cot_instruction: str = "Please reason step by step, and put your final answer within \\boxed{answer}."
    format_template: str = "{instruction}\n{question}"
    use_chat_template: bool = False  # Set True for instruction-tuned models (e.g. Llama-3.1-8B-Instruct)


@dataclass
class SoftPromptConfig:
    """Soft prompt memory parameterization configuration."""
    num_tokens: int = 20
    init_strategy: Literal["question_embedding", "random"] = "question_embedding"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"


@dataclass
class LoRAConfig:
    """LoRA memory parameterization configuration (Ablation)."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    learning_rate: float = 1e-4


@dataclass
class PrefixTuningConfig:
    """Prefix tuning memory parameterization configuration (Ablation)."""
    num_prefix_tokens: int = 20
    learning_rate: float = 1e-3


@dataclass
class GRPOConfig:
    """Group Relative Policy Optimization (GRPO) training configuration."""
    group_size: int = 8                # K = 8 rollouts per problem
    train_steps: int = 10              # S = 10 gradient steps per sample (or 2 for ablation)
    max_length: int = 2048             # Maximum sequence length
    temperature: float = 0.8           # 0.8 for Qwen, 0.7 for Llama
    top_p: float = 0.95
    kl_penalty_beta: float = 0.01      # β = 0.01
    clip_range: float = 0.2
    loss_variant: str = "DR-GRPO"      # DR-GRPO loss variant
    batch_size: int = 64
    gradient_accumulation_steps: int = 1


@dataclass
class RetrievalConfig:
    """Memory key construction and similarity retrieval configuration."""
    retrieval_mode: Literal["input", "kv"] = "input"  # ELM_input vs ELM_kv
    top_k: int = 1                                     # Paper uses top-1 retrieval
    similarity_metric: str = "cosine"
    normalize_embeddings: bool = True


@dataclass
class VerifierConfig:
    """Response selection and PRM verifier configuration."""
    model_name_or_path: str = "Qwen/Qwen2.5-Math-PRM-7B"
    enabled: bool = True
    device: str = "auto"
    use_mock_fallback: bool = True  # Fallback to step-heuristic verifier if PRM weights not local


@dataclass
class ELMConfig:
    """Unified master configuration for ELM."""
    model_name: str = "Qwen/Qwen2.5-Math-7B"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42
    
    # Sub-configurations
    prompt: PromptConfig = field(default_factory=PromptConfig)
    soft_prompt: SoftPromptConfig = field(default_factory=SoftPromptConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    prefix_tuning: PrefixTuningConfig = field(default_factory=PrefixTuningConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    
    # Memory type
    memory_type: Literal["soft_prompt", "lora", "prefix_tuning", "full_model"] = "soft_prompt"
    
    # Benchmark settings
    benchmark_name: Literal["AMC23", "AIME24", "MATH500", "mock"] = "AMC23"
    data_split: str = "test"
    output_dir: str = "./outputs/elm_experiments"

