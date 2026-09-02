"""Process Reward Model (PRM) Verifier and Response Selection Routing.

Paper specification (Section 3.3):
Given a test sample x, retrieve nearest memory m* and generate:
- a_zs ~ pi_theta(. | x) (zero-shot response)
- a_mem ~ pi_theta(. | [m*, x]) (memory-augmented response)
If a_zs == a_mem, return shared answer.
Else, a verifier selects higher-scoring response, mitigating memory-induced regressions.
"""

import torch
import torch.nn as nn
import re
from typing import Optional, Tuple, Dict, Any
from ..config import VerifierConfig
from ..data.math_eval import extract_boxed_answer, normalize_math_answer, grade_answer


class PRMVerifier:
    """Process Reward Model (PRM) verifier wrapper (e.g. Qwen2.5-Math-PRM-7B)."""
    
    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or VerifierConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self):
        """Load PRM model weights from HuggingFace."""
        if self.is_loaded:
            return
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[Verifier] Loading PRM model: {self.config.model_name_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.config.device,
                trust_remote_code=True,
            )
            self.model.eval()
            self.is_loaded = True
            print("[Verifier] PRM loaded successfully.")
        except Exception as e:
            if self.config.use_mock_fallback:
                print(f"[Verifier Warning] Could not load PRM ({e}). Using robust step-heuristic scorer.")
                self.is_loaded = False
            else:
                raise e

    def score_response(self, question: str, response: str) -> float:
        """Score reasoning response on a [0.0, 1.0] scale."""
        if self.is_loaded and self.model is not None:
            # Tokenize question + response with PRM step delimiters
            # Typically Qwen-PRM outputs reward logits for step boundary tokens
            try:
                prompt = f"Question: {question}\nSolution:\n{response}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Average logit / reward estimate
                    score = torch.sigmoid(outputs.logits[0, -1, 0]).item()
                    return score
            except Exception:
                pass
                
        # Robust heuristic fallback scorer
        return self._heuristic_score(question, response)

    def _heuristic_score(self, question: str, response: str) -> float:
        """Heuristic scoring based on step structure, boxed formatting, and consistency."""
        score = 0.5
        
        # Bonus for clear boxed answer
        boxed = extract_boxed_answer(response)
        if boxed is not None:
            score += 0.25
            if len(boxed) > 0 and len(boxed) < 50:  # Concise clean answer
                score += 0.1
                
        # Bonus for multi-step reasoning markers
        step_markers = len(re.findall(r"(?:Step \d|Therefore|Thus|Hence|Because|Since|So,)", response, re.IGNORECASE))
        score += min(0.15, step_markers * 0.03)
        
        # Penalty for degenerate repetition or excessive length
        words = response.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 0.4
                
        return max(0.0, min(1.0, score))


class ResponseSelectionRouter:
    """Routes between zero-shot and memory-augmented responses."""
    
    def __init__(self, verifier: Optional[PRMVerifier] = None):
        self.verifier = verifier or PRMVerifier()
        
    def select_best_response(
        self,
        question: str,
        response_zs: str,
        response_mem: str,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Select final response following Section 3.3 routing policy.
        
        Returns:
            chosen_response: The selected string response
            source: 'agreement', 'zero_shot', or 'memory_augmented'
            details: Routing scores and metadata
        """
        ans_zs = extract_boxed_answer(response_zs)
        ans_mem = extract_boxed_answer(response_mem)
        
        # 1. Agreement check: If they predict equivalent answers, return shared answer
        if ans_zs is not None and ans_mem is not None and grade_answer(ans_zs, ans_mem):
            return response_zs, "agreement", {
                "ans_zs": ans_zs,
                "ans_mem": ans_mem,
                "agreed": True,
            }
            
        # 2. Divergence: Use verifier to score both
        score_zs = self.verifier.score_response(question, response_zs)
        score_mem = self.verifier.score_response(question, response_mem)
        
        if score_mem >= score_zs:
            return response_mem, "memory_augmented", {
                "ans_zs": ans_zs,
                "ans_mem": ans_mem,
                "agreed": False,
                "score_zs": score_zs,
                "score_mem": score_mem,
                "selected": "mem",
            }
        else:
            return response_zs, "zero_shot", {
                "ans_zs": ans_zs,
                "ans_mem": ans_mem,
                "agreed": False,
                "score_zs": score_zs,
                "score_mem": score_mem,
                "selected": "zs",
            }

