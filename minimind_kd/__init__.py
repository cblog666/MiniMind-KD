"""MiniMind-KD: a small, readable Kimi-K3 × DeepSeek-V4 research model."""

from .config import ModelConfig, TrainConfig
from .modeling.model import MiniMindKDForCausalLM

__all__ = ["MiniMindKDForCausalLM", "ModelConfig", "TrainConfig"]
__version__ = "0.1.0"
