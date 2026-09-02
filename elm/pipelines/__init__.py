"""Pipelines reproducing all evaluation protocols and baselines from the ELM paper."""

from .online_elm import OnlineELMPipeline
from .offline_elm import OfflineELMPipeline
from .cross_transfer import CrossDatasetTransferPipeline

__all__ = [
    "OnlineELMPipeline",
    "OfflineELMPipeline",
    "CrossDatasetTransferPipeline",
]

