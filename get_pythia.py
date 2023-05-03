from transformers import AutoConfig
path = "EleutherAI/gpt-neo-2.7B"
config = AutoConfig.from_pretrained(path if path is not None else "EleutherAI/pythia-70m-deduped")
