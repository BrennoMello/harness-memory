"""Self-supervised internal reward computation via Majority Voting (TTRL).

Paper Formula (Eq. 2):
Given a question x, the model generates N candidate responses {y_1, ..., y_N},
and a consensus answer y* is derived via majority voting:
R(y_i, y*) = 1 if parsed(y_i) == y* else 0.
"""

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from ..data.math_eval import extract_boxed_answer, normalize_math_answer, grade_answer


@dataclass
class RewardResult:
    """Internal reward result computed across candidate rollouts."""
    consensus_answer: Optional[str]
    rewards: List[float]
    parsed_answers: List[Optional[str]]
    vote_distribution: Dict[str, int]
    agreement_count: int
    total_rollouts: int


def extract_majority_vote_consensus(
    responses: List[str],
) -> Tuple[Optional[str], int, Dict[str, int], List[Optional[str]]]:
    """Extract answers from responses and identify majority voting consensus y*."""
    parsed_answers: List[Optional[str]] = []
    normalized_votes: List[str] = []
    norm_to_raw: Dict[str, str] = {}
    
    for resp in responses:
        ans = extract_boxed_answer(resp)
        parsed_answers.append(ans)
        if ans is not None:
            norm_ans = normalize_math_answer(ans)
            if norm_ans:
                normalized_votes.append(norm_ans)
                if norm_ans not in norm_to_raw:
                    norm_to_raw[norm_ans] = ans
                    
    if not normalized_votes:
        return None, 0, {}, parsed_answers
        
    counts = Counter(normalized_votes)
    most_common_norm, highest_count = counts.most_common(1)[0]
    consensus_raw = norm_to_raw[most_common_norm]
    
    vote_distribution = {norm_to_raw.get(k, k): v for k, v in counts.items()}
    return consensus_raw, highest_count, vote_distribution, parsed_answers


def compute_internal_rewards(
    responses: List[str],
    consensus_answer: Optional[str] = None,
) -> RewardResult:
    """Compute binary internal reward R(y_i, y*) for each rollout response."""
    if consensus_answer is None:
        consensus_answer, agree_count, vote_dist, parsed_answers = extract_majority_vote_consensus(responses)
    else:
        parsed_answers = [extract_boxed_answer(r) for r in responses]
        vote_dist = {}
        agree_count = sum(1 for a in parsed_answers if grade_answer(a, consensus_answer))
        
    rewards: List[float] = []
    for parsed in parsed_answers:
        if consensus_answer is not None and grade_answer(parsed, consensus_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return RewardResult(
        consensus_answer=consensus_answer,
        rewards=rewards,
        parsed_answers=parsed_answers,
        vote_distribution=vote_dist,
        agreement_count=agree_count,
        total_rollouts=len(responses),
    )

