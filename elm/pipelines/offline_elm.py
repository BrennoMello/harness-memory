"""Offline Memory Evaluation Pipeline (Section 4.1 & Table 1).

Trains individual memories per sample, evaluates each memory across the full evaluation set,
and computes Oracle ZS vs Mem, Negative Flip Rate (1 -> 0), and Memory Overlap (IoU).
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm

from ..config import ELMConfig
from ..data.benchmarks import ProblemInstance
from ..data.math_eval import extract_boxed_answer, grade_answer
from ..memory.soft_prompt import SoftPromptMemory
from ..training.internal_reward import compute_internal_rewards
from ..training.grpo import GRPOTrainer


@dataclass
class OfflineMemoryEvalResult:
    """Evaluation result for a single trained memory over a benchmark."""
    memory_id: str
    source_sample_id: str
    zero_shot_accuracy: float
    memory_accuracy: float
    oracle_accuracy: float
    positive_flips: int        # 0 -> 1 (improved)
    negative_flips: int        # 1 -> 0 (regressed)
    neutral_count: int         # unchanged
    improved_sample_ids: List[str]
    regressed_sample_ids: List[str]


class OfflineELMPipeline:
    """Pipeline for offline per-sample memory training and benchmark evaluation."""
    
    def __init__(
        self,
        config: Optional[ELMConfig] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config or ELMConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.grpo_trainer = GRPOTrainer(
            config=self.config.grpo,
            learning_rate=self.config.soft_prompt.learning_rate,
            device=self.device,
        )

    def _format_prompt(self, question: str) -> str:
        prompt_text = self.config.prompt.format_template.format(
            instruction=self.config.prompt.cot_instruction,
            question=question,
        )
        if self.config.prompt.use_chat_template and self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt_text

    def train_memory_for_sample(self, sample: ProblemInstance) -> SoftPromptMemory:
        """Train a single soft prompt memory on a given sample."""
        prompt_text = self._format_prompt(sample.question)
        
        if self.model is None or self.tokenizer is None:
            # Fallback mock memory
            return SoftPromptMemory(num_tokens=self.config.soft_prompt.num_tokens, hidden_dim=4096, device=self.device)
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        embed_layer = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else self.model.model.embed_tokens
        with torch.no_grad():
            init_embeds = embed_layer(inputs["input_ids"])
            
        memory = SoftPromptMemory(
            num_tokens=self.config.soft_prompt.num_tokens,
            hidden_dim=init_embeds.size(-1),
            init_embeddings=init_embeds,
            device=self.device,
        )
        
        # Sample rollouts
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                do_sample=True,
                temperature=self.config.grpo.temperature,
                num_return_sequences=self.config.grpo.group_size,
                max_new_tokens=1024,
            )
        rollouts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        rewards_res = compute_internal_rewards(rollouts)
        
        # Train with GRPO
        self.grpo_trainer.train_soft_prompt_memory(
            memory=memory,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            rollout_texts=rollouts,
            rewards=rewards_res.rewards,
            train_steps=self.config.grpo.train_steps,
        )
        return memory

    def evaluate_memory(
        self,
        memory: SoftPromptMemory,
        eval_dataset: List[ProblemInstance],
        source_sample_id: str = "",
    ) -> OfflineMemoryEvalResult:
        """Evaluate a single memory unit across the entire evaluation set."""
        pos_flips = 0
        neg_flips = 0
        neutral = 0
        zs_correct = 0
        mem_correct = 0
        oracle_correct = 0
        
        improved_ids = []
        regressed_ids = []
        
        for sample in eval_dataset:
            prompt_text = self._format_prompt(sample.question)
            
            if self.model is not None and self.tokenizer is not None:
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
                
                # Zero-shot greedy
                with torch.no_grad():
                    out_zs = self.model.generate(input_ids=inputs["input_ids"], do_sample=False, max_new_tokens=1024)
                resp_zs = self.tokenizer.decode(out_zs[0], skip_special_tokens=True)
                
                # Memory-augmented greedy
                embed_layer = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else self.model.model.embed_tokens
                token_embeds = embed_layer(inputs["input_ids"])
                ext_embeds, ext_mask = memory.forward_embeddings(token_embeds, inputs.get("attention_mask"))
                with torch.no_grad():
                    out_mem = self.model.generate(inputs_embeds=ext_embeds, attention_mask=ext_mask, do_sample=False, max_new_tokens=1024)
                resp_mem = self.tokenizer.decode(out_mem[0], skip_special_tokens=True)
            else:
                resp_zs = "Final Answer: \\boxed{1}"
                resp_mem = "Final Answer: \\boxed{1}"
                
            ans_zs = extract_boxed_answer(resp_zs)
            ans_mem = extract_boxed_answer(resp_mem)
            
            c_zs = grade_answer(ans_zs, sample.ground_truth_answer)
            c_mem = grade_answer(ans_mem, sample.ground_truth_answer)
            
            if c_zs:
                zs_correct += 1
            if c_mem:
                mem_correct += 1
            if c_zs or c_mem:
                oracle_correct += 1
                
            if not c_zs and c_mem:
                pos_flips += 1
                improved_ids.append(sample.id)
            elif c_zs and not c_mem:
                neg_flips += 1
                regressed_ids.append(sample.id)
            else:
                neutral += 1
                
        n = len(eval_dataset)
        return OfflineMemoryEvalResult(
            memory_id=f"mem_{source_sample_id}",
            source_sample_id=source_sample_id,
            zero_shot_accuracy=(zs_correct / n) * 100.0 if n else 0.0,
            memory_accuracy=(mem_correct / n) * 100.0 if n else 0.0,
            oracle_accuracy=(oracle_correct / n) * 100.0 if n else 0.0,
            positive_flips=pos_flips,
            negative_flips=neg_flips,
            neutral_count=neutral,
            improved_sample_ids=improved_ids,
            regressed_sample_ids=regressed_ids,
        )

    @staticmethod
    def compute_iou(set_a: List[str], set_b: List[str]) -> float:
        """Compute Intersection over Union (IoU) between improved sample sets."""
        s_a = set(set_a)
        s_b = set(set_b)
        union = len(s_a.union(s_b))
        if union == 0:
            return 0.0
        return len(s_a.intersection(s_b)) / union

