import torch
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, AutoConfig, GPT2LMHeadModel
from .utils import print_main
from pathlib import Path
from transformers.modeling_utils import no_init_weights
from accelerate import init_empty_weights
from typing import Optional

LANGUAGE_MODELS = [
    "gptj",
    "pythia"
]


def gptj_config():
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.num_layers = 28
    config.num_heads = 16
    config.hidden_size = 256 * config.num_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True
    config.gradient_checkpointing = True
    return config


def get_gptj(
    gradient_checkpointing: bool = True,
    from_pretrained=False,
    init_weights=True,
) -> torch.nn.Module:
    """
    Loads GPTJ language model from HF
    """
    print_main("Loading GPTJ language model...")
    config = gptj_config()
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"
    if from_pretrained:
        raise NotImplemented("GPTJ pretrained not implemented")
    else:
        with no_init_weights():
            #init_empty_weights(not init_weights):  # no_init_weights():
            model = GPTNeoForCausalLM(config=config)
    return model

def pythia_config(path: Optional[str] = None):
    config = AutoConfig.from_pretrained(path if path is not None else "EleutherAI/pythia-70m-deduped")
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True
    config.gradient_checkpointing = True
    return config

def get_pythia(
    path: str = None,
    gradient_checkpointing: bool = True,
    init_weights=True,
    # from_pretrained=True,  # TODO think about this
) -> torch.nn.Module:
    """
    Loads Pythia language model from HF
    """
    print_main("Loading Pythia language model...")
    config = pythia_config(path)
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"

    with no_init_weights(not init_weights):
        # TODO better internet connection
        model = GPTNeoXForCausalLM.from_pretrained(config._name_or_path, config=config, ignore_mismatched_sizes=True, force_download=True)
    return model

