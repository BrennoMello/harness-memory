"""Command Line Interface (CLI) for running ELM pipelines and baselines."""

import argparse
import sys
import os
import json
import torch

from .config import ELMConfig
from .data.benchmarks import load_benchmark
from .pipelines.online_elm import OnlineELMPipeline
from .pipelines.offline_elm import OfflineELMPipeline
from .pipelines.cross_transfer import CrossDatasetTransferPipeline
from .pipelines.baselines.zero_shot import ZeroShotCoTPipeline
from .pipelines.baselines.icl import InContextLearningPipeline
from .pipelines.baselines.reasoning_bank import ReasoningBankPipeline
from .pipelines.baselines.lag import LAGPipeline
from .pipelines.baselines.verifier_at_2 import VerifierAt2Pipeline
from .evaluator import BenchmarkSummary, BenchmarkEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ELM: Continual Self-Improvement with Lightweight Experiential Latent Memories (arXiv:2606.17803)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command to execute")
    
    # Online streaming ELM (Algorithm 1)
    online_parser = subparsers.add_parser("online", help="Run Online Streaming ELM (Algorithm 1)")
    online_parser.add_argument("--benchmark", type=str, default="AMC23", choices=["AMC23", "AIME24", "MATH500", "mock"])
    online_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B")
    online_parser.add_argument("--retrieval", type=str, default="input", choices=["input", "kv"])
    online_parser.add_argument("--max_samples", type=int, default=None)
    online_parser.add_argument("--output_dir", type=str, default="./outputs/online")
    online_parser.add_argument("--mock_run", action="store_true", help="Run in mock mode without downloading model weights")
    
    # Offline ELM (Table 1)
    offline_parser = subparsers.add_parser("offline", help="Run Offline Per-Sample Memory Evaluation (Table 1)")
    offline_parser.add_argument("--benchmark", type=str, default="AMC23", choices=["AMC23", "AIME24", "MATH500", "mock"])
    offline_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B")
    offline_parser.add_argument("--max_samples", type=int, default=None)
    offline_parser.add_argument("--mock_run", action="store_true")
    
    # Cross-dataset transfer (Figure 3)
    transfer_parser = subparsers.add_parser("transfer", help="Run Cross-Dataset Memory Transfer (Figure 3)")
    transfer_parser.add_argument("--source", type=str, default="AMC23", choices=["AMC23", "AIME24", "mock"])
    transfer_parser.add_argument("--target", type=str, default="AIME24", choices=["AMC23", "AIME24", "mock"])
    transfer_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B")
    transfer_parser.add_argument("--mock_run", action="store_true")
    
    # Baselines
    baseline_parser = subparsers.add_parser("baseline", help="Run baseline evaluation")
    baseline_parser.add_argument("--type", type=str, required=True, choices=["zero_shot", "icl", "reasoning_bank", "lag", "verifier@2"])
    baseline_parser.add_argument("--benchmark", type=str, default="AMC23", choices=["AMC23", "AIME24", "MATH500", "mock"])
    baseline_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B")
    baseline_parser.add_argument("--max_samples", type=int, default=None)
    baseline_parser.add_argument("--mock_run", action="store_true")
    
    return parser


def load_model_and_tokenizer(model_name: str, mock_run: bool = False):
    """Load model and tokenizer or return None for mock mode."""
    if mock_run:
        print("[CLI] Running in MOCK mode (No weights loaded).")
        return None, None
        
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[CLI] Loading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model.eval()
        print("[CLI] Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"[CLI Warning] Failed to load model {model_name}: {e}. Falling back to mock mode.")
        return None, None


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    config = ELMConfig()
    
    if args.command == "online":
        config.retrieval.retrieval_mode = args.retrieval
        dataset = load_benchmark(args.benchmark, max_samples=args.max_samples)
        model, tokenizer = load_model_and_tokenizer(args.model, args.mock_run)
        
        pipeline = OnlineELMPipeline(config=config, model=model, tokenizer=tokenizer)
        results = pipeline.run_stream(dataset)
        
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, f"online_elm_{args.benchmark}_{args.retrieval}.json")
        with open(summary_path, "w") as f:
            json.dump({
                "benchmark": args.benchmark,
                "model": args.model,
                "retrieval": args.retrieval,
                "accuracy": results["accuracy"],
                "total": results["total_samples"],
            }, f, indent=2)
        print(f"\n[Result Saved] -> {summary_path}")
        
    elif args.command == "offline":
        dataset = load_benchmark(args.benchmark, max_samples=args.max_samples)
        model, tokenizer = load_model_and_tokenizer(args.model, args.mock_run)
        pipeline = OfflineELMPipeline(config=config, model=model, tokenizer=tokenizer)
        
        print(f"\n[ELM Offline] Training per-sample memories on {len(dataset)} examples...")
        # Train on first sample as showcase
        if dataset:
            mem = pipeline.train_memory_for_sample(dataset[0])
            eval_res = pipeline.evaluate_memory(mem, dataset, source_sample_id=dataset[0].id)
            print(f"\n[Offline Results for Memory {dataset[0].id}]")
            print(f"Zero-Shot Acc: {eval_res.zero_shot_accuracy:.2f}%")
            print(f"Memory Acc: {eval_res.memory_accuracy:.2f}%")
            print(f"Oracle ZS vs Mem Acc: {eval_res.oracle_accuracy:.2f}%")
            print(f"Positive Flips (0->1): {eval_res.positive_flips} | Negative Flips (1->0): {eval_res.negative_flips}")
            
    elif args.command == "transfer":
        src_ds = load_benchmark(args.source, max_samples=10 if args.mock_run else None)
        tgt_ds = load_benchmark(args.target, max_samples=10 if args.mock_run else None)
        model, tokenizer = load_model_and_tokenizer(args.model, args.mock_run)
        
        transfer_pipe = CrossDatasetTransferPipeline(config=config, model=model, tokenizer=tokenizer)
        src_pool = transfer_pipe.train_source_memories(src_ds)
        res = transfer_pipe.evaluate_transfer(tgt_ds, src_pool)
        print(f"\n[Transfer Complete] Transfer Gain: {res['gain']:+.2f}%")
        
    elif args.command == "baseline":
        dataset = load_benchmark(args.benchmark, max_samples=args.max_samples)
        model, tokenizer = load_model_and_tokenizer(args.model, args.mock_run)
        
        if args.type == "zero_shot":
            pipe = ZeroShotCoTPipeline(model=model, tokenizer=tokenizer)
            res = pipe.evaluate(dataset)
        elif args.type == "icl":
            pipe = InContextLearningPipeline(model=model, tokenizer=tokenizer)
            res = pipe.evaluate_stream(dataset)
        elif args.type == "reasoning_bank":
            pipe = ReasoningBankPipeline(model=model, tokenizer=tokenizer)
            res = pipe.evaluate_stream(dataset)
        elif args.type == "lag":
            pipe = LAGPipeline(model=model, tokenizer=tokenizer)
            res = pipe.evaluate_stream(dataset)
        elif args.type == "verifier@2":
            pipe = VerifierAt2Pipeline(model=model, tokenizer=tokenizer)
            res = pipe.evaluate(dataset)
            
        print(f"\n[Baseline {args.type}] Accuracy: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")


if __name__ == "__main__":
    main()

