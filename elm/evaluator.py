"""Evaluation metrics and benchmark result summarizer."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class BenchmarkSummary:
    """Consolidated benchmark summary."""
    benchmark_name: str
    method_name: str
    model_name: str
    total_samples: int
    accuracy_pass1: float
    correct_count: int
    oracle_accuracy: Optional[float] = None
    negative_flip_rate: Optional[float] = None
    positive_flip_rate: Optional[float] = None
    memory_overlap_iou: Optional[float] = None
    average_retrieval_similarity: Optional[float] = None
    agreement_rate: Optional[float] = None
    trainable_params_mb: Optional[float] = None


class BenchmarkEvaluator:
    """Calculates all metrics reported in the ELM paper."""
    
    @staticmethod
    def compute_accuracy(correct: int, total: int) -> float:
        return (correct / total) * 100.0 if total > 0 else 0.0

    @staticmethod
    def compute_negative_flip_rate(neg_flips: int, pos_flips: int) -> float:
        """Paper footnote 3: Flip rate is defined as (# negative flips) / (# negative flips + # positive flips)."""
        total_flips = neg_flips + pos_flips
        if total_flips == 0:
            return 0.0
        return (neg_flips / total_flips) * 100.0

    @staticmethod
    def compute_iou(samples_a: List[str], samples_b: List[str]) -> float:
        """Intersection over Union (IoU) between improved sample IDs."""
        set_a = set(samples_a)
        set_b = set(samples_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return len(set_a & set_b) / union

    @staticmethod
    def format_results_table(summaries: List[BenchmarkSummary]) -> str:
        """Format markdown results table matching paper format."""
        headers = ["Benchmark", "Method", "Model", "Greedy Pass@1 (%)", "Oracle (%)", "Neg Flip Rate (%)"]
        rows = []
        for s in summaries:
            rows.append([
                s.benchmark_name,
                s.method_name,
                s.model_name,
                f"{s.accuracy_pass1:.1f}%",
                f"{s.oracle_accuracy:.1f}%" if s.oracle_accuracy is not None else "-",
                f"{s.negative_flip_rate:.1f}%" if s.negative_flip_rate is not None else "-",
            ])
            
        col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
        
        header_line = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        body_lines = [
            "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
            for row in rows
        ]
        return "\n".join([header_line, sep_line] + body_lines)

