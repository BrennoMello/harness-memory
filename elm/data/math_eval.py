"""Mathematical answer extraction, normalization, and verification logic."""

import re
from typing import Optional, Any


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content inside the last \\boxed{...} in a model completion.
    
    Handles nested braces properly by tracking brace depth.
    """
    if not text:
        return None
        
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # Fallback: look for "The answer is" or "Final Answer:"
        patterns = [
            r"(?:the\s+final\s+answer\s+is|final\s+answer:?|the\s+answer\s+is)\s*[:\$]?\s*([^\n\.$]+)",
            r"\\boxed\s*([0-9a-zA-Z\+\-\*/\^]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    # Found \boxed{, now find matching closing brace
    start_idx = idx + len("\\boxed{")
    depth = 1
    i = start_idx
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start_idx : i - 1].strip()
    return text[start_idx:].strip()


def normalize_math_answer(answer: Optional[str]) -> str:
    """Normalize mathematical expression for reliable string and numeric equivalence."""
    if answer is None:
        return ""
    
    text = str(answer).strip()
    
    # Strip common markdown / math symbols
    text = text.replace("$", "").replace("\\%", "%")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\text{", "").replace("\\mathrm{", "")
    text = text.replace("\\,", "").replace("\\ ", " ")
    
    # Strip spaces around operators
    text = re.sub(r"\s+", " ", text).strip()
    
    # Standardize fractions: \frac{a}{b} -> a/b
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\dfrac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    
    # Remove leading/trailing formatting
    text = text.rstrip(".")
    text = text.strip()
    
    # Try Sympy evaluation for exact numeric / symbolic simplification if available
    try:
        import sympy
        parsed = sympy.sympify(text)
        return str(sympy.simplify(parsed))
    except Exception:
        pass
        
    return text.lower()


def grade_answer(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """Check if predicted answer matches ground truth answer mathematically."""
    if predicted is None or ground_truth is None:
        return False
        
    norm_pred = normalize_math_answer(predicted)
    norm_gt = normalize_math_answer(ground_truth)
    
    if not norm_pred or not norm_gt:
        return False
        
    if norm_pred == norm_gt:
        return True
        
    # Numeric equivalence fallback
    try:
        val_pred = float(eval(norm_pred))
        val_gt = float(eval(norm_gt))
        if abs(val_pred - val_gt) < 1e-5:
            return True
    except Exception:
        pass
        
    # Direct string containment / equality
    return norm_pred.strip() == norm_gt.strip()

