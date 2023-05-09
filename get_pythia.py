from transformers import AutoConfig
path = "gpt2"
config = AutoConfig.from_pretrained(path if path is not None else "EleutherAI/pythia-70m-deduped")
