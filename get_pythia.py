from transformers import AutoConfig
path = "EleutherAI/pythia-410m-deduped"
config = AutoConfig.from_pretrained(path if path is not None else "EleutherAI/pythia-70m-deduped")
