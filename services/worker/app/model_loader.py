"""
Model loader for LLM inference with KV cache integration.
Supports HuggingFace Transformers models (LLaMA, GPT-2, etc.)
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages LLM model loading and initialization.

    Features:
    - Lazy loading: Model loaded on first inference request
    - GPU support: Automatically uses CUDA if available
    - HuggingFace integration: Compatible with any AutoModelForCausalLM
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.config: Optional[Any] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: Optional[str] = None
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.head_dim: int = 0
        logger.info(f"ModelLoader initialized with device: {self.device}")

    def load_model(self, model_name: str = "gpt2") -> Dict[str, Any]:
        """
        Load a HuggingFace model for inference.

        Args:
            model_name: Model identifier (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")

        Returns:
            Dict with model info: num_layers, num_heads, head_dim, device
        """
        if self.model is not None and self.model_name == model_name:
            logger.info(f"Model {model_name} already loaded")
            return self._get_model_info()

        try:
            logger.info(f"Loading model: {model_name} on {self.device}")

            # Load config first to get architecture details
            self.config = AutoConfig.from_pretrained(model_name)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()  # Set to evaluation mode
            self.model_name = model_name

            # Extract architecture details
            self._extract_model_architecture()

            logger.info(
                f"Model loaded successfully: {self.num_layers} layers, "
                f"{self.num_heads} heads, {self.head_dim} head_dim"
            )

            return self._get_model_info()

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _extract_model_architecture(self):
        """Extract number of layers, heads, and head dimension from config."""
        # Different models use different config keys
        config = self.config

        # Number of layers
        if hasattr(config, "num_hidden_layers"):
            self.num_layers = config.num_hidden_layers
        elif hasattr(config, "n_layer"):
            self.num_layers = config.n_layer
        else:
            self.num_layers = 12  # Default fallback

        # Number of attention heads
        if hasattr(config, "num_attention_heads"):
            self.num_heads = config.num_attention_heads
        elif hasattr(config, "n_head"):
            self.num_heads = config.n_head
        else:
            self.num_heads = 8  # Default fallback

        # Head dimension
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
            self.head_dim = hidden_size // self.num_heads
        else:
            self.head_dim = 64  # Default fallback

    def _get_model_info(self) -> Dict[str, Any]:
        """Return model architecture information."""
        return {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "device": self.device,
            "dtype": (
                str(next(self.model.parameters()).dtype) if self.model else "unknown"
            ),
        }

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text to tensor."""
        if self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        return tokens.input_ids.to(self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)

    def generate_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Run one forward pass (generate one token).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            past_key_values: Previous KV cache as tuple of layer key-value pairs

        Returns:
            (logits, past_key_values as tuple)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Handle both tuple and DynamicCache outputs for compatibility
        past_kv = outputs.past_key_values
        if past_kv is not None and hasattr(past_kv, "to_legacy_cache"):
            # Convert DynamicCache to tuple if needed
            past_kv = past_kv.to_legacy_cache()

        return outputs.logits, past_kv

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            logger.info(f"Unloading model: {self.model_name}")
            del self.model
            del self.tokenizer
            del self.config
            self.model = None
            self.tokenizer = None
            self.config = None
            self.model_name = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage statistics."""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "device": self.device,
            }
        return {"device": "cpu", "allocated_gb": 0, "reserved_gb": 0}
