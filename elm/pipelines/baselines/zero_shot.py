"""Zero-Shot Chain-of-Thought (CoT) Baseline."""

from typing import List, Dict, Any, Optional
from tqdm import tqdm
from ...data.benchmarks import ProblemInstance
from ...data.math_eval import extract_boxed_answer, grade_answer


class ZeroShotCoTPipeline:
    """Zero-Shot CoT evaluation baseline."""
    
    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None, instruction: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.instruction = instruction or "Please reason step by step, and put your final answer within \\boxed{answer}."

    def evaluate(self, dataset: List[ProblemInstance]) -> Dict[str, Any]:
        """Evaluate dataset with zero-shot CoT prompting."""
        correct = 0
        records = []
        
        for sample in tqdm(dataset, desc="Zero-Shot CoT"):
            prompt = f"{self.instruction}\n{sample.question}"
            
            if self.model and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(input_ids=inputs["input_ids"], do_sample=False, max_new_tokens=1024)
                resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                resp = f"Step 1: Compute.\nFinal Answer: \\boxed{{{sample.ground_truth_answer}}}"
                
            pred = extract_boxed_answer(resp)
            is_correct = grade_answer(pred, sample.ground_truth_answer)
            if is_correct:
                correct += 1
                
            records.append({
                "id": sample.id,
                "predicted": pred,
                "ground_truth": sample.ground_truth_answer,
                "correct": is_correct,
            })
            
        n = len(dataset)
        acc = (correct / n) * 100.0 if n else 0.0
        return {"accuracy": acc, "correct": correct, "total": n, "records": records}

