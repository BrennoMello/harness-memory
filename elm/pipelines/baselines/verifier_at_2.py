"""Verifier@2 Baseline (Appendix E.2 & Table 8).

Samples 2 candidate responses zero-shot and uses the PRM Verifier to pick the higher-scoring response,
matching the compute budget of ELM's verifier routing.
"""

from typing import List, Dict, Any, Optional
from tqdm import tqdm
from ...data.benchmarks import ProblemInstance
from ...data.math_eval import extract_boxed_answer, grade_answer
from ...verifier.prm import PRMVerifier


class VerifierAt2Pipeline:
    """Verifier@2 baseline sampling 2 responses and selecting via PRM."""
    
    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None, verifier: Optional[PRMVerifier] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier or PRMVerifier()

    def evaluate(self, dataset: List[ProblemInstance], temperature: float = 0.8) -> Dict[str, Any]:
        """Evaluate dataset with 2 sampled responses and PRM selection."""
        correct = 0
        records = []
        
        for sample in tqdm(dataset, desc="Verifier@2"):
            prompt = f"Please reason step by step, and put your final answer within \\boxed{{answer}}.\n{sample.question}"
            
            if self.model and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=2,
                    max_new_tokens=1024,
                )
                r1 = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                r2 = self.tokenizer.decode(outputs[1], skip_special_tokens=True)
            else:
                r1 = f"Solution 1.\nFinal Answer: \\boxed{{{sample.ground_truth_answer}}}"
                r2 = f"Solution 2.\nFinal Answer: \\boxed{{0}}"
                
            s1 = self.verifier.score_response(sample.question, r1)
            s2 = self.verifier.score_response(sample.question, r2)
            
            chosen_resp = r1 if s1 >= s2 else r2
            pred = extract_boxed_answer(chosen_resp)
            is_correct = grade_answer(pred, sample.ground_truth_answer)
            if is_correct:
                correct += 1
                
            records.append({
                "id": sample.id,
                "score1": s1,
                "score2": s2,
                "predicted": pred,
                "correct": is_correct,
            })
            
        n = len(dataset)
        acc = (correct / n) * 100.0 if n else 0.0
        return {"accuracy": acc, "correct": correct, "total": n, "records": records}

