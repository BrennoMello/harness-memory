"""Algorithm 1: Continual Self-Improvement with Experiential Latent Memories (ELM).

Implements the strictly online streaming protocol where the model encounters problems
one at a time, accumulates lightweight modular memories, and retrieves them for future problems.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..config import ELMConfig
from ..data.benchmarks import ProblemInstance
from ..data.math_eval import extract_boxed_answer, grade_answer
from ..memory.soft_prompt import SoftPromptMemory
from ..memory.pool import MemoryPool, MemoryEntry
from ..memory.indexing import MemoryIndex
from ..training.internal_reward import compute_internal_rewards, extract_majority_vote_consensus
from ..training.grpo import GRPOTrainer
from ..verifier.prm import PRMVerifier, ResponseSelectionRouter


@dataclass
class OnlineStepResult:
    """Detailed record of a single step in the streaming online ELM protocol."""
    step_idx: int
    problem_id: str
    question: str
    ground_truth_answer: str
    predicted_answer: Optional[str]
    is_correct: bool
    retrieved_memory_id: Optional[str]
    retrieval_similarity: Optional[float]
    response_source: str                 # 'zero_shot', 'agreement', 'memory_augmented'
    zero_shot_response: str
    memory_augmented_response: Optional[str]
    consensus_answer: Optional[str]
    majority_vote_count: int
    pool_size_after: int
    training_loss: Optional[float]


class OnlineELMPipeline:
    """Streaming Online Continual Self-Improvement Pipeline (Algorithm 1)."""
    
    def __init__(
        self,
        config: Optional[ELMConfig] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        verifier: Optional[PRMVerifier] = None,
    ):
        self.config = config or ELMConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
        self.memory_pool = MemoryPool()
        self.index = MemoryIndex(retrieval_mode=self.config.retrieval.retrieval_mode)
        self.grpo_trainer = GRPOTrainer(
            config=self.config.grpo,
            learning_rate=self.config.soft_prompt.learning_rate,
            device=self.device,
        )
        self.verifier = verifier or PRMVerifier(self.config.verifier)
        self.router = ResponseSelectionRouter(self.verifier)
        
    def _format_prompt(self, question: str) -> str:
        """Format problem with zero-shot CoT instruction."""
        prompt_text = self.config.prompt.format_template.format(
            instruction=self.config.prompt.cot_instruction,
            question=question,
        )
        if self.config.prompt.use_chat_template and self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt_text

    def _generate_response(
        self,
        prompt_text: str,
        memory_module: Optional[SoftPromptMemory] = None,
        greedy: bool = True,
        num_samples: int = 1,
        temperature: float = 0.8,
    ) -> List[str]:
        """Generate response(s) with optional prepended soft prompt memory."""
        if self.model is None or self.tokenizer is None:
            # Mock generator fallback for fast CPU testing / verification
            mock_ans = "42"
            return [f"Reasoning step by step.\nFinal Answer: \\boxed{{{mock_ans}}}"] * num_samples
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": 1024,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = self.config.grpo.top_p
            gen_kwargs["num_return_sequences"] = num_samples

        with torch.no_grad():
            if memory_module is not None:
                # Prepend soft prompt embeddings
                embed_layer = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else self.model.model.embed_tokens
                token_embeds = embed_layer(inputs["input_ids"])
                extended_embeds, extended_mask = memory_module.forward_embeddings(
                    token_embeds, inputs.get("attention_mask")
                )
                outputs = self.model.generate(
                    inputs_embeds=extended_embeds,
                    attention_mask=extended_mask,
                    **gen_kwargs,
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_kwargs,
                )
                
        responses = []
        for out in outputs:
            # Decode generated tokens
            resp_text = self.tokenizer.decode(out, skip_special_tokens=True)
            responses.append(resp_text)
        return responses

    def run_stream(self, stream: List[ProblemInstance]) -> Dict[str, Any]:
        """Execute Algorithm 1 over an incoming stream of problems."""
        results: List[OnlineStepResult] = []
        correct_count = 0
        
        print(f"\n[ELM Online] Starting streaming run over {len(stream)} problems...")
        
        for step_idx, sample in enumerate(tqdm(stream, desc="ELM Streaming")):
            prompt_text = self._format_prompt(sample.question)
            
            # --- 1. Memory Retrieval ---
            retrieved_entry: Optional[MemoryEntry] = None
            sim_score: Optional[float] = None
            
            if len(self.memory_pool) > 0:
                if self.model is not None and self.tokenizer is not None:
                    q_key = MemoryIndex.extract_input_query(self.model, self.tokenizer, prompt_text, self.device)
                else:
                    q_key = torch.randn(4096, device=self.device)
                    
                matches = self.index.retrieve(self.memory_pool, query_input_key=q_key, top_k=1)
                if matches:
                    retrieved_entry, sim_score = matches[0]
                    
            # --- 2. Response Generation & Routing ---
            resp_zs = self._generate_response(prompt_text, memory_module=None, greedy=True)[0]
            
            if retrieved_entry is not None and retrieved_entry.memory_module is not None:
                resp_mem = self._generate_response(
                    prompt_text, memory_module=retrieved_entry.memory_module, greedy=True
                )[0]
                final_resp, source, _ = self.router.select_best_response(
                    sample.question, resp_zs, resp_mem
                )
            else:
                final_resp = resp_zs
                resp_mem = None
                source = "zero_shot"
                
            pred_ans = extract_boxed_answer(final_resp)
            is_correct = grade_answer(pred_ans, sample.ground_truth_answer)
            if is_correct:
                correct_count += 1
                
            # --- 3. Memory Creation (TTRL + GRPO) ---
            # Sample K=8 rollouts for majority voting internal reward
            rollouts = self._generate_response(
                prompt_text,
                memory_module=None,
                greedy=False,
                num_samples=self.config.grpo.group_size,
                temperature=self.config.grpo.temperature,
            )
            reward_res = compute_internal_rewards(rollouts)
            
            # Initialize new soft prompt memory from problem embeddings
            if self.model is not None and self.tokenizer is not None:
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
                embed_layer = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else self.model.model.embed_tokens
                with torch.no_grad():
                    init_embeds = embed_layer(inputs["input_ids"])
                hidden_dim = init_embeds.size(-1)
            else:
                init_embeds = torch.randn(len(prompt_text.split()), 4096, device=self.device)
                hidden_dim = 4096
                
            new_memory = SoftPromptMemory(
                num_tokens=self.config.soft_prompt.num_tokens,
                hidden_dim=hidden_dim,
                init_embeddings=init_embeds,
                device=self.device,
            )
            
            # Train memory with GRPO if rollouts available
            train_loss = None
            if self.model is not None and self.tokenizer is not None:
                train_info = self.grpo_trainer.train_soft_prompt_memory(
                    memory=new_memory,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt_text=prompt_text,
                    rollout_texts=rollouts,
                    rewards=reward_res.rewards,
                    train_steps=self.config.grpo.train_steps,
                )
                train_loss = train_info.get("final_loss")
                
            # Compute input key and add to memory pool M
            mem_key = new_memory.get_input_key()
            mem_entry = MemoryEntry(
                memory_id=f"mem_{sample.id}_{step_idx}",
                source_sample_id=sample.id,
                source_question=sample.question,
                consensus_answer=reward_res.consensus_answer,
                input_key=mem_key,
                memory_module=new_memory,
                created_at_step=step_idx,
                metadata={"vote_count": reward_res.agreement_count},
            )
            self.memory_pool.add(mem_entry)
            
            # Record step result
            step_record = OnlineStepResult(
                step_idx=step_idx,
                problem_id=sample.id,
                question=sample.question,
                ground_truth_answer=sample.ground_truth_answer,
                predicted_answer=pred_ans,
                is_correct=is_correct,
                retrieved_memory_id=retrieved_entry.memory_id if retrieved_entry else None,
                retrieval_similarity=sim_score,
                response_source=source,
                zero_shot_response=resp_zs,
                memory_augmented_response=resp_mem,
                consensus_answer=reward_res.consensus_answer,
                majority_vote_count=reward_res.agreement_count,
                pool_size_after=len(self.memory_pool),
                training_loss=train_loss,
            )
            results.append(step_record)
            
        accuracy = (correct_count / len(stream)) * 100.0 if stream else 0.0
        print(f"[ELM Online] Completed. Final Accuracy (greedy pass@1): {accuracy:.2f}% ({correct_count}/{len(stream)})")
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_samples": len(stream),
            "step_results": results,
            "final_pool_size": len(self.memory_pool),
        }

