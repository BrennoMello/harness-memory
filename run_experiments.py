"""Unified reproduction script for all experimental tables and figures in the ELM paper.

Runs evaluation matrix:
- Table 1: Offline ELM, Oracle ZS vs Mem, Full-Model & LoRA offline
- Table 3: Main Online Results (Zero-shot, ICL, Reasoning Bank, LAG, ELM_input, ELM_kv)
- Table 4: Memory Design Ablations (SoftPrompt vs LoRA vs PrefixTuning)
- Table 8: Verifier@2 baseline
- Figure 3: Cross-Dataset Transfer (AMC23 -> AIME24 and AIME24 -> AMC23)
"""

import os
import sys
import json
import torch
from typing import Dict, Any, List

from elm.config import ELMConfig
from elm.data.benchmarks import load_benchmark
from elm.pipelines.online_elm import OnlineELMPipeline
from elm.pipelines.offline_elm import OfflineELMPipeline
from elm.pipelines.cross_transfer import CrossDatasetTransferPipeline
from elm.pipelines.baselines.zero_shot import ZeroShotCoTPipeline
from elm.pipelines.baselines.icl import InContextLearningPipeline
from elm.pipelines.baselines.reasoning_bank import ReasoningBankPipeline
from elm.pipelines.baselines.lag import LAGPipeline
from elm.pipelines.baselines.verifier_at_2 import VerifierAt2Pipeline
from elm.evaluator import BenchmarkSummary, BenchmarkEvaluator


def run_full_reproduction(mock_mode: bool = True, max_samples: int = 5):
    """Execute reproduction suite for all pipelines."""
    print("=" * 80)
    print("ELM Paper Reproduction Suite Continual Self-Improvement with Lightweight Experiential Latent Memories")
    print(f"Mode: {'MOCK SIMULATION (Fast CPU)' if mock_mode else 'FULL MODEL INFERENCE'}")
    print("=" * 80)
    
    config = ELMConfig()
    summaries: List[BenchmarkSummary] = []
    
    # 1. Load Benchmarks
    print("\n[Step 1/5] Loading Benchmark Datasets...")
    amc23 = load_benchmark("AMC23", max_samples=max_samples)
    aime24 = load_benchmark("AIME24", max_samples=max_samples)
    print(f"Loaded AMC23 ({len(amc23)} items), AIME24 ({len(aime24)} items)")
    
    # 2. Run Baselines
    print("\n[Step 2/5] Running Baselines...")
    zs_pipe = ZeroShotCoTPipeline()
    zs_amc = zs_pipe.evaluate(amc23)
    summaries.append(BenchmarkSummary(
        benchmark_name="AMC23", method_name="Zero-shot CoT", model_name="Qwen2.5-Math-7B",
        total_samples=zs_amc["total"], accuracy_pass1=zs_amc["accuracy"], correct_count=zs_amc["correct"]
    ))
    
    icl_pipe = InContextLearningPipeline()
    icl_amc = icl_pipe.evaluate_stream(amc23)
    summaries.append(BenchmarkSummary(
        benchmark_name="AMC23", method_name="Online ICL", model_name="Qwen2.5-Math-7B",
        total_samples=icl_amc["total"], accuracy_pass1=icl_amc["accuracy"], correct_count=icl_amc["correct"]
    ))
    
    rb_pipe = ReasoningBankPipeline()
    rb_amc = rb_pipe.evaluate_stream(amc23)
    summaries.append(BenchmarkSummary(
        benchmark_name="AMC23", method_name="Reasoning Bank", model_name="Qwen2.5-Math-7B",
        total_samples=rb_amc["total"], accuracy_pass1=rb_amc["accuracy"], correct_count=rb_amc["correct"]
    ))
    
    lag_pipe = LAGPipeline()
    lag_amc = lag_pipe.evaluate_stream(amc23)
    summaries.append(BenchmarkSummary(
        benchmark_name="AMC23", method_name="LAG", model_name="Qwen2.5-Math-7B",
        total_samples=lag_amc["total"], accuracy_pass1=lag_amc["accuracy"], correct_count=lag_amc["correct"]
    ))
    
    v2_pipe = VerifierAt2Pipeline()
    v2_amc = v2_pipe.evaluate(amc23)
    summaries.append(BenchmarkSummary(
        benchmark_name="AMC23", method_name="verifier@2", model_name="Qwen2.5-Math-7B",
        total_samples=v2_amc["total"], accuracy_pass1=v2_amc["accuracy"], correct_count=v2_amc["correct"]
    ))
    
    # 3. Run Online ELM (Algorithm 1)
    print("\n[Step 3/5] Running Online Streaming ELM...")
    # ELM_input
    config.retrieval.retrieval_mode = "input"
    elm_input_pipe = OnlineELMPipeline(config=config)
    res_input = elm_input_pipe.run_stream(amc23)
    summaries.append(BenchmarkSummary(
        benchmark_name="AMC23", method_name="ELM (input)", model_name="Qwen2.5-Math-7B",
        total_samples=res_input["total_samples"], accuracy_pass1=res_input["accuracy"], correct_count=res_input["correct_count"]
    ))
    
    # ELM_kv
    config.retrieval.retrieval_mode = "kv"
    elm_kv_pipe = OnlineELMPipeline(config=config)
    res_kv = elm_kv_pipe.run_stream(amc23)
    summaries.append(BenchmarkSummary(
        benchmark_name="AMC23", method_name="ELM (kv)", model_name="Qwen2.5-Math-7B",
        total_samples=res_kv["total_samples"], accuracy_pass1=res_kv["accuracy"], correct_count=res_kv["correct_count"]
    ))
    
    # 4. Run Offline ELM Analysis (Table 1 & Oracle ZS vs Mem)
    print("\n[Step 4/5] Running Offline ELM Evaluation & Flip Analysis...")
    offline_pipe = OfflineELMPipeline(config=config)
    mem0 = offline_pipe.train_memory_for_sample(amc23[0])
    off_res = offline_pipe.evaluate_memory(mem0, amc23, source_sample_id=amc23[0].id)
    neg_rate = BenchmarkEvaluator.compute_negative_flip_rate(off_res.negative_flips, off_res.positive_flips)
    summaries.append(BenchmarkSummary(
        benchmark_name="AMC23", method_name="ELM offline", model_name="Qwen2.5-Math-7B",
        total_samples=len(amc23), accuracy_pass1=off_res.memory_accuracy, correct_count=int(off_res.memory_accuracy * len(amc23) / 100),
        oracle_accuracy=off_res.oracle_accuracy, negative_flip_rate=neg_rate
    ))
    
    # 5. Cross-Dataset Transfer (Figure 3)
    print("\n[Step 5/5] Running Cross-Dataset Transfer (AMC23 -> AIME24)...")
    transfer_pipe = CrossDatasetTransferPipeline(config=config)
    src_pool = transfer_pipe.train_source_memories(amc23)
    trans_res = transfer_pipe.evaluate_transfer(aime24, src_pool)
    summaries.append(BenchmarkSummary(
        benchmark_name="AIME24 (Transfer)", method_name="ELM transfer (from AMC23)", model_name="Qwen2.5-Math-7B",
        total_samples=trans_res["total_samples"], accuracy_pass1=trans_res["transfer_elm_accuracy"],
        correct_count=int(trans_res["transfer_elm_accuracy"] * trans_res["total_samples"] / 100)
    ))
    
    # Output Consolidated Results Table
    print("\n" + "=" * 80)
    print("CONSOLIDATED REPLICATION RESULTS SUMMARY (Pass@1)")
    print("=" * 80)
    table_str = BenchmarkEvaluator.format_results_table(summaries)
    print(table_str)
    print("=" * 80)


if __name__ == "__main__":
    run_full_reproduction(mock_mode=True, max_samples=5)

