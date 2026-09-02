"""Cross-Dataset Memory Transfer Pipeline (Section 5 & Figure 3).

Evaluates whether memories pre-trained on Dataset A (e.g., AMC23) transfer
and improve zero-shot reasoning performance on Dataset B (e.g., AIME24).
"""

from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch

from ..config import ELMConfig
from ..data.benchmarks import ProblemInstance
from ..data.math_eval import extract_boxed_answer, grade_answer
from ..memory.pool import MemoryPool
from ..memory.indexing import MemoryIndex
from ..pipelines.online_elm import OnlineELMPipeline
from ..verifier.prm import PRMVerifier, ResponseSelectionRouter


class CrossDatasetTransferPipeline:
    """Pipeline for evaluating cross-dataset transfer of learned latent memories."""
    
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
        self.index = MemoryIndex(retrieval_mode=self.config.retrieval.retrieval_mode)
        self.verifier = verifier or PRMVerifier(self.config.verifier)
        self.router = ResponseSelectionRouter(self.verifier)
        self.online_pipe = OnlineELMPipeline(config=self.config, model=model, tokenizer=tokenizer, verifier=self.verifier)

    def train_source_memories(self, source_dataset: List[ProblemInstance]) -> MemoryPool:
        """Train and collect memory pool M on source dataset."""
        print(f"[Transfer] Training memories on source dataset ({len(source_dataset)} problems)...")
        res = self.online_pipe.run_stream(source_dataset)
        return self.online_pipe.memory_pool

    def evaluate_transfer(
        self,
        target_dataset: List[ProblemInstance],
        pre_trained_pool: MemoryPool,
    ) -> Dict[str, Any]:
        """Evaluate pre-trained memories on target dataset using retrieval + verifier routing."""
        print(f"[Transfer] Evaluating on target dataset ({len(target_dataset)} problems) with {len(pre_trained_pool)} memories...")
        
        zs_correct = 0
        elm_correct = 0
        details = []
        
        for sample in tqdm(target_dataset, desc="Evaluating Transfer"):
            prompt_text = self.online_pipe._format_prompt(sample.question)
            
            # 1. Retrieve most similar memory from pre-trained pool
            if self.model is not None and self.tokenizer is not None:
                q_key = MemoryIndex.extract_input_query(self.model, self.tokenizer, prompt_text, self.device)
            else:
                q_key = torch.randn(4096, device=self.device)
                
            matches = self.index.retrieve(pre_trained_pool, query_input_key=q_key, top_k=1)
            retrieved_entry = matches[0][0] if matches else None
            
            # 2. Generate responses
            resp_zs = self.online_pipe._generate_response(prompt_text, memory_module=None, greedy=True)[0]
            
            if retrieved_entry and retrieved_entry.memory_module:
                resp_mem = self.online_pipe._generate_response(
                    prompt_text, memory_module=retrieved_entry.memory_module, greedy=True
                )[0]
                final_resp, source, _ = self.router.select_best_response(
                    sample.question, resp_zs, resp_mem
                )
            else:
                final_resp = resp_zs
                source = "zero_shot"
                
            ans_zs = extract_boxed_answer(resp_zs)
            ans_final = extract_boxed_answer(final_resp)
            
            c_zs = grade_answer(ans_zs, sample.ground_truth_answer)
            c_elm = grade_answer(ans_final, sample.ground_truth_answer)
            
            if c_zs:
                zs_correct += 1
            if c_elm:
                elm_correct += 1
                
            details.append({
                "problem_id": sample.id,
                "zero_shot_correct": c_zs,
                "transfer_elm_correct": c_elm,
                "retrieved_memory_id": retrieved_entry.memory_id if retrieved_entry else None,
                "source": source,
            })
            
        n = len(target_dataset)
        zs_acc = (zs_correct / n) * 100.0 if n else 0.0
        elm_acc = (elm_correct / n) * 100.0 if n else 0.0
        
        print(f"[Transfer Results] Zero-Shot Acc: {zs_acc:.2f}% | Transfer ELM Acc: {elm_acc:.2f}% (Delta: {elm_acc - zs_acc:+.2f}%)")
        
        return {
            "zero_shot_accuracy": zs_acc,
            "transfer_elm_accuracy": elm_acc,
            "gain": elm_acc - zs_acc,
            "total_samples": n,
            "details": details,
        }

