# PaliGemma: A Multimodal Vision-Language Transformer

This project implements a **multimodal vision-language model** called **PaliGemma** using PyTorch. The model processes both **images and text** to perform tasks such as **image captioning** or **visual question answering**.

## 🔍 Overview

- **Vision Encoder**: Based on **SigLIP**, a contrastive vision model using a Vision Transformer (ViT) backbone.
- **Language Decoder**: A **transformer-based decoder** called **Gemma**.
- **Training Paradigm**: Follows **contrastive learning** principles (like CLIP) to align image and text embeddings.
- **Inference Capabilities**: Supports generation using **temperature, top-p sampling**, and **KV caching** for efficient decoding.

---

## 🧠 Architecture Details

### Vision Encoder (SigLIP)
- Splits input images into non-overlapping patches.
- Applies positional embeddings, multi-head self-attention, and feed-forward networks (FFNs).
- Produces a global image embedding.
- Trained using **contrastive loss** to align with text.

### Language Decoder (Gemma)
- Transformer decoder using:
  - **Rotary positional embeddings**
  - **Grouped Query Attention** for memory efficiency
  - **RMSNorm** for stability
  - **KV Cache** for fast autoregressive inference
- Receives the image embedding as a prefix to the text input.

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/PaliGemma.git
cd PaliGemma
```
## 2. Install Dependencies
```bash
pip install -r requirements.txt
```
## 3. Download Model Weights
Place the downloaded HuggingFace-compatible model:

## 4. Prepare Input
Place your test image as image.png in the root directory (or change the path in launch_inference.sh).

Edit the PROMPT in launch_inference.sh to customize the prompt.

##5. Run Inference
```bash
./launch_inference.sh
