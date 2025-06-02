from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Check if model_path exists
    print(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found.")

    # Load the tokenizer manually
    tokenizer_path = os.path.join(model_path, "tokenizer.json")  # or "vocab.txt" based on your model
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    
    tokenizer_path = model_path.replace("\\", "/")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    if len(safetensors_files) == 0:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device=device) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load model config
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Initialize the model
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the model state_dict
    model.load_state_dict(tensors, strict=False)

    # Tie model weights
    model.tie_weights()

    return model, tokenizer
