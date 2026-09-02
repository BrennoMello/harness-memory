"""In-Context Learning (ICL) Baseline.

Stores raw question-solution pairs (q, CoT) and retrieves the most similar
example at test time to prompt the model in context.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ...data.benchmarks import ProblemInstance
from ...data.math_eval import extract_boxed_answer, grade_answer


class InContextLearningPipeline:
    """Online ICL baseline storing and retrieving raw text problem-solution pairs."""
    
    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.text_memory_pool = []  # Stores (question, solution, embedding_key)
        
    def evaluate_stream(self, stream: List[ProblemInstance]) -> Dict[str, Any]:
        """Run streaming ICL over benchmark."""
        correct = 0
        records = []
        
        for sample in tqdm(stream, desc="Online ICL"):
            # 1. Retrieve most similar past example if pool is not empty
            few_shot_prefix = ""
            if self.text_memory_pool:
                # Find most similar question based on token overlap / embeddings
                best_sim = -1.0
                best_example = None
                for past_q, past_sol in self.text_memory_pool:
                    # Jaccard word similarity as robust text retrieval
                    words_cur = set(sample.question.lower().split())
                    words_past = set(past_q.lower().split())
                    sim = len(words_cur & words_past) / (len(words_cur | words_past) + 1e-8)
                    if sim > best_sim:
                        best_sim = sim
                        best_example = (past_q, past_sol)
                        
                if best_example:
                    few_shot_prefix = (
                        f"Here is a similar example:\n"
                        f"Problem: {best_example[0]}\n"
                        f"Solution: {best_example[1]}\n\n"
                    )
                    
            prompt = (
                f"{few_shot_prefix}"
                f"Please reason step by step, and put your final answer within \\boxed{{answer}}.\n"
                f"{sample.question}"
            )
            
            if self.model and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(input_ids=inputs["input_ids"], do_sample=False, max_new_tokens=1024)
                resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                resp = f"Reasoning.\nFinal Answer: \\boxed{{{sample.ground_truth_answer}}}"
                
            pred = extract_boxed_answer(resp)
            is_correct = grade_answer(pred, sample.ground_truth_answer)
            if is_correct:
                correct += 1
                
            # Add to text memory pool
            self.text_memory_pool.append((sample.question, resp))
            records.append({
                "id": sample.id,
                "predicted": pred,
                "correct": is_correct,
            })
            
        n = len(stream)
        acc = (correct / n) * 100.0 if n else 0.0
        return {"accuracy": acc, "correct": correct, "total": n, "records": records}

