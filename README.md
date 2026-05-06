# 🌟 PaliGemma: Custom PyTorch Inference

A from-scratch, custom PyTorch implementation for running inference on **PaliGemma**, Google's open-weights Vision-Language Model (VLM). 

This project breaks down the PaliGemma architecture into its core foundational components, allowing you to understand exactly how images and text are stitched together, how the KV-cache is managed, and how autoregressive generation works without relying on the black-box abstractions of high-level libraries.

## 🧠 Architecture Overview

PaliGemma connects a Vision Transformer to a Large Language Model:
1. **Vision Encoder (SigLIP)**: Processes images into patch embeddings natively.
2. **Text Decoder (Gemma)**: A causal language model using Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), and RMSNorm.
3. **Multimodal Projector**: A linear layer projecting the SigLIP visual embeddings into the Gemma text embedding space.

## 📂 Project Structure

* **`launch_inference.sh`**: The simple bash script entry point to configure generation parameters and run the model.
* **`inference.py`**: The core execution engine. Handles the autoregressive generation loop, token sampling (Top-P and Temperature), and KV-Cache management.
* **`modeling_gemma.py`**: Deep dive into the Gemma LLM architecture. Includes RoPE, RMSNorm, Multi-Layer Perceptrons, and the wrapper VLM class `PaliGemmaForConditionalGeneration`.
* **`modelling_siglip.py`**: The implementation of the SigLIP Vision Transformer.
* **`processing_paligemma.py`**: Combines image resizing, pixel normalization, and text tokenization. Automates the prepending of `<image>` tokens to your prompt.
* **`utils.py`**: Handles loading local HuggingFace `.safetensors` weights and matching them to our custom PyTorch architecture.

## 🚀 Getting Started

### 1. Set up the Environment
Create a virtual environment and install the required dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Model Weights
Because this is a low-level implementation, it expects the raw model weights saved locally.
1. Go to the [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) repository on HuggingFace.
2. Download the `.safetensors` files, `config.json`, and `tokenizer.json` into a local folder (e.g., `paligemma-weights/`).

### 3. Provide a Test Image
Place an image in the project root directory and name it `image.png` (or update the shell script to point to your desired visual input).

### 4. Configure & Run
Open `launch_inference.sh` and update the `MODEL_PATH` variable to point to where you downloaded the HuggingFace weights. 

```bash
# Example update in launch_inference.sh
MODEL_PATH="./paligemma-weights"
ONLY_CPU="True" # Set to True if testing on a Mac without CUDA 
```

Then, run the inference instance:
```bash
bash launch_inference.sh
```

## ⚙️ Generation Parameters

You can freely tweak the generation parameters inside `launch_inference.sh`:
* `PROMPT`: The textual prompt or question.
* `MAX_TOKENS_TO_GENERATE`: Stop generating after this many tokens.
* `TEMPERATURE`: The randomness of the generation (lower is strictly more factual, higher is more creative).
* `TOP_P`: Nucleus sampling threshold.
* `DO_SAMPLE`: Set to `"False"` for greedy decoding, or `"True"` to enable Top-P & Temperature sampling.
