# from transformers import AutoConfig
# path = "ViT-H-14"
# config = AutoConfig.from_pretrained(path if path is not None else "EleutherAI/pythia-70m-deduped")

import open_clip
name, pretrained = "ViT-H-14", "laion2b_s32b_b79k"
device = "cuda"
encoder = open_clip.create_model(name, device=device, precision="fp16" if "cuda" in str(device) else "fp32", pretrained=pretrained).visual

while True:continue