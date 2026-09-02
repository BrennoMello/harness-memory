"""Unit and integration test suite for ELM replication."""

import os
import tempfile
import torch
import pytest

from elm.data.math_eval import extract_boxed_answer, normalize_math_answer, grade_answer
from elm.data.benchmarks import load_benchmark, ProblemInstance
from elm.memory.soft_prompt import SoftPromptMemory
from elm.memory.pool import MemoryPool, MemoryEntry
from elm.memory.indexing import MemoryIndex
from elm.training.internal_reward import extract_majority_vote_consensus, compute_internal_rewards
from elm.training.grpo import GRPOTrainer
from elm.verifier.prm import PRMVerifier, ResponseSelectionRouter
from elm.pipelines.online_elm import OnlineELMPipeline


def test_math_eval_boxed_extraction():
    # Nested braces
    text1 = "After computing, the answer is \\boxed{\\frac{3}{4}}."
    assert extract_boxed_answer(text1) == "\\frac{3}{4}"
    
    # Simple integer
    text2 = "Therefore, we get \\boxed{42}."
    assert extract_boxed_answer(text2) == "42"
    
    # Fallback pattern
    text3 = "Final Answer: 130"
    assert extract_boxed_answer(text3) == "130"


def test_grade_answer_equivalence():
    assert grade_answer("\\frac{6}{8}", "\\frac{3}{4}") is True
    assert grade_answer("42", "42.0") is True
    assert grade_answer("x + y", "y + x") is True
    assert grade_answer("10", "11") is False


def test_soft_prompt_memory():
    num_tokens = 20
    hidden_dim = 128
    batch_size = 2
    seq_len = 10
    
    mem = SoftPromptMemory(num_tokens=num_tokens, hidden_dim=hidden_dim)
    input_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.ones(batch_size, seq_len)
    
    ext_embeds, ext_mask = mem.forward_embeddings(input_embeds, mask)
    
    assert ext_embeds.shape == (batch_size, num_tokens + seq_len, hidden_dim)
    assert ext_mask.shape == (batch_size, num_tokens + seq_len)
    
    key = mem.get_input_key()
    assert key.shape == (hidden_dim,)


def test_internal_reward_majority_voting():
    responses = [
        "Step by step. \\boxed{9}",
        "Calculation... \\boxed{9}",
        "Another way. \\boxed{9}",
        "Wrong method. \\boxed{5}",
    ]
    res = compute_internal_rewards(responses)
    assert res.consensus_answer == "9"
    assert res.agreement_count == 3
    assert res.rewards == [1.0, 1.0, 1.0, 0.0]


def test_grpo_advantages():
    trainer = GRPOTrainer()
    rewards = [1.0, 1.0, 0.0, 0.0]
    adv = trainer.compute_group_advantages(rewards)
    assert adv.shape == (4,)
    assert adv[0] > 0
    assert adv[2] < 0


def test_memory_pool_and_indexing():
    pool = MemoryPool()
    index = MemoryIndex(retrieval_mode="input")
    
    key1 = torch.tensor([1.0, 0.0, 0.0])
    key2 = torch.tensor([0.0, 1.0, 0.0])
    
    pool.add(MemoryEntry(
        memory_id="m1", source_sample_id="s1", source_question="Q1",
        consensus_answer="A1", input_key=key1
    ))
    pool.add(MemoryEntry(
        memory_id="m2", source_sample_id="s2", source_question="Q2",
        consensus_answer="A2", input_key=key2
    ))
    
    query = torch.tensor([0.9, 0.1, 0.0])
    matches = index.retrieve(pool, query, top_k=1)
    
    assert len(matches) == 1
    assert matches[0][0].memory_id == "m1"
    assert matches[0][1] > 0.8


def test_memory_pool_persistence():
    pool = MemoryPool()
    mem = SoftPromptMemory(num_tokens=5, hidden_dim=16)
    pool.add(MemoryEntry(
        memory_id="m_test", source_sample_id="s_test", source_question="Q?",
        consensus_answer="42", input_key=mem.get_input_key(), memory_module=mem
    ))
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        pool.save_to_disk(tmp_path)
        new_pool = MemoryPool()
        new_pool.load_from_disk(tmp_path)
        assert len(new_pool) == 1
        assert new_pool.entries[0].memory_id == "m_test"
        assert new_pool.entries[0].consensus_answer == "42"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_response_selection_router():
    verifier = PRMVerifier()
    router = ResponseSelectionRouter(verifier)
    
    # Matching answers -> agreement
    resp_zs = "Final Answer: \\boxed{10}"
    resp_mem = "Solution: \\boxed{10}"
    chosen, source, details = router.select_best_response("What is 5+5?", resp_zs, resp_mem)
    assert source == "agreement"
    assert details["agreed"] is True
    
    # Diverging answers -> verifier scoring
    resp_zs_bad = "No steps. Answer: 10"
    resp_mem_good = "Step 1: 5+5=10. Therefore \\boxed{10}"
    chosen2, source2, details2 = router.select_best_response("What is 5+5?", resp_zs_bad, resp_mem_good)
    assert source2 in ["zero_shot", "memory_augmented"]


def test_online_elm_pipeline_mock():
    dataset = load_benchmark("MOCK", max_samples=3)
    pipeline = OnlineELMPipeline()
    results = pipeline.run_stream(dataset)
    
    assert results["total_samples"] == 3
    assert results["final_pool_size"] == 3
    assert len(results["step_results"]) == 3

