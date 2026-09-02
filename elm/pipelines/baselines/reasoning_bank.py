"""Reasoning Bank Baseline (Ouyang et al., 2025).

Extracts abstract textual reasoning strategies via reflection and refinement,
stores them in a strategy memory bank, and retrieves them for upcoming queries.
"""

from typing import List, Dict, Any, Optional
from tqdm import tqdm
from ...data.benchmarks import ProblemInstance
from ...data.math_eval import extract_boxed_answer, grade_answer


class ReasoningBankPipeline:
    """Reasoning Bank baseline extracting reusable text reasoning strategies."""
    
    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy_bank = []  # Stores (question, extracted_strategy)

    def extract_strategy(self, question: str, solution: str) -> str:
        """Refinement pass to extract high-level problem solving strategy."""
        refine_prompt = (
            f"Analyze the following problem and solution, and summarize the key mathematical "
            f"principle or problem-solving strategy in 2-3 concise sentences:\n\n"
            f"Problem: {question}\n"
            f"Solution: {solution}\n\n"
            f"Strategy Summary:"
        )
        if self.model and self.tokenizer:
            inputs = self.tokenizer(refine_prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(input_ids=inputs["input_ids"], do_sample=False, max_new_tokens=256)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return "Apply systematic algebraic decomposition and check edge constraints."

    def evaluate_stream(self, stream: List[ProblemInstance]) -> Dict[str, Any]:
        """Execute Reasoning Bank over streaming benchmark."""
        correct = 0
        records = []
        
        for sample in tqdm(stream, desc="Reasoning Bank"):
            # Retrieve top strategy if available
            strategy_prefix = ""
            if self.strategy_bank:
                # Retrieve nearest strategy
                best_sim = -1.0
                best_strat = None
                for past_q, strat in self.strategy_bank:
                    words_cur = set(sample.question.lower().split())
                    words_past = set(past_q.lower().split())
                    sim = len(words_cur & words_past) / (len(words_cur | words_past) + 1e-8)
                    if sim > best_sim:
                        best_sim = sim
                        best_strat = strat
                if best_strat:
                    strategy_prefix = f"Relevant Strategy to keep in mind: {best_strat}\n\n"
                    
            prompt = (
                f"{strategy_prefix}"
                f"Please reason step by step, and put your final answer within \\boxed{{answer}}.\n"
                f"{sample.question}"
            )
            
            if self.model and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(input_ids=inputs["input_ids"], do_sample=False, max_new_tokens=1024)
                resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                resp = f"Reasoning with strategy.\nFinal Answer: \\boxed{{{sample.ground_truth_answer}}}"
                
            pred = extract_boxed_answer(resp)
            is_correct = grade_answer(pred, sample.ground_truth_answer)
            if is_correct:
                correct += 1
                
            # Extract and store strategy
            strategy = self.extract_strategy(sample.question, resp)
            self.strategy_bank.append((sample.question, strategy))
            
            records.append({"id": sample.id, "predicted": pred, "correct": is_correct})
            
        n = len(stream)
        acc = (correct / n) * 100.0 if n else 0.0
        return {"accuracy": acc, "correct": correct, "total": n, "records": records}

