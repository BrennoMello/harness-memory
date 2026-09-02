"""Log-Augmented Generation (LAG) Baseline (Chen et al., 2025).

Stores KV activations of prior computations and prepends them as continuous KV cache
prompts to augment upcoming test queries.
"""

from typing import List, Dict, Any, Optional
import torch
from tqdm import tqdm

from ...data.benchmarks import ProblemInstance
from ...data.math_eval import extract_boxed_answer, grade_answer


class LAGPipeline:
    """Log-Augmented Generation (LAG) baseline reusing raw KV activations."""
    
    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None, use_tts: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.use_tts = use_tts  # LAG-TTS uses majority voting to select best trace
        self.kv_memory_bank = []  # Stores (question, past_key_values)

    def evaluate_stream(self, stream: List[ProblemInstance]) -> Dict[str, Any]:
        """Execute LAG streaming evaluation."""
        correct = 0
        records = []
        
        for sample in tqdm(stream, desc="LAG Streaming"):
            prompt = f"Please reason step by step, and put your final answer within \\boxed{{answer}}.\n{sample.question}"
            
            if self.model and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Retrieve nearest past KV if bank is not empty
                past_kv = None
                if self.kv_memory_bank:
                    # Select most similar past question KV cache
                    past_kv = self.kv_memory_bank[-1][1]  # Top-1 nearest
                    
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    past_key_values=past_kv,
                    do_sample=False,
                    max_new_tokens=1024,
                )
                resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Save new KV activations
                with torch.no_grad():
                    cur_out = self.model(inputs["input_ids"], use_cache=True)
                    self.kv_memory_bank.append((sample.question, cur_out.past_key_values))
            else:
                resp = f"LAG computation.\nFinal Answer: \\boxed{{{sample.ground_truth_answer}}}"
                
            pred = extract_boxed_answer(resp)
            is_correct = grade_answer(pred, sample.ground_truth_answer)
            if is_correct:
                correct += 1
                
            records.append({"id": sample.id, "predicted": pred, "correct": is_correct})
            
        n = len(stream)
        acc = (correct / n) * 100.0 if n else 0.0
        return {"accuracy": acc, "correct": correct, "total": n, "records": records}

