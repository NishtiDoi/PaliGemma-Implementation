from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    """
    Recursively moves dictionary values (tensors) to the specified computational device.

    Args:
        model_inputs (dict): Dictionary containing the input tensors for the model.
        device (str): The target device (e.g., 'mps', 'cuda', 'cpu').

    Returns:
        dict: The same dictionary with all tensors moved to the target device.
    """
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    """
    Prepares raw text and image data for model inference by transforming them into tensors.

    This function handles the end-to-end preprocessing pipeline: loading the image,
    tokenizing the text, normalizing pixel values, and ensuring all tensors are 
    located on the correct hardware device.

    Args:
        processor (PaliGemmaProcessor): The processor responsible for tokenization 
            and image transformation.
        prompt (str): The text instruction or question for the model.
        image_file_path (str): Local path to the image file.
        device (str): The target device for inference (e.g., 'mps', 'cuda', or 'cpu').

    Returns:
        dict: A dictionary containing 'input_ids', 'pixel_values', and 'attention_mask' 
              tensors mapped to the specified device.
    """
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """
    Executes the autoregressive generation loop to produce text from an image and prompt.

    The function passes inputs through the model, manages the KV-Cache for efficiency,
    samples the next token based on logits, and decodes the final sequence into a string.

    Args:
        model (PaliGemmaForConditionalGeneration): The loaded VLM model. the brain of the operation, responsible for generating text based on the input image and prompt.
        processor (PaliGemmaProcessor): Processor for encoding inputs and decoding tokens. the translator
        device (str): Computation hardware in use.
        prompt (str): User-provided text prompt.
        image_file_path (str): Path to the input image.
        max_tokens_to_generate (int): Limit for the output length.
        temperature (float): Scaling factor for token probability distribution.
        top_p (float): Threshold for nucleus sampling.
        do_sample (bool): If True, uses probabilistic sampling; otherwise, uses greedy search.
    """
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"] # In your code, a 1 tells the model "this is a real word/image pixel, look at it." A 0 (if present) would tell the model "this is just empty padding to fill space, ignore it."
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id #In your loop, the code checks: "Did the model just output the stop_token?" If yes, the loop breaks, and the model stops talking.
    generated_tokens = [] #  This lsit will hold the tokens that the model generates in response to the prompt and image. Each time the model produces a new token, it's added to this list. At the end of the loop, all generated tokens are combined and decoded into a human-readable string.

    for _ in range(max_tokens_to_generate): # Autoregressive gen loop, the model "speaks" here
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        ) # sends current input tokens, image pixels, attention mask, and the KV-Cache to the model. 
          # The model processes this information and produces the next token's logits and an updated KV-Cache for the next iteration.
        
        kv_cache = outputs["kv_cache"] # After the model generates the next token, it also updates the KV-Cache with new key-value pairs that will be used in the next iteration of the loop. 
                                       # This allows the model to maintain context and generate coherent responses without having to reprocess all previous tokens and image features from scratch.
        next_token_logits = outputs["logits"][:, -1, :] 

        if do_sample: # If do_sample is True, the code applies temperature scaling to the logits to control randomness and then uses nucleus (top-p) sampling to select the next token.
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else: # uses greedy decoding, which simply selects the token with the highest probability (the argmax) as the next token in the sequence.
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # the below code does shap control + bookkeeping
        assert next_token.size() == (1, 1) # (sanity checkpoint) guarantees single batch and single token output 
        next_token = next_token.squeeze(0) # removes the batch dimension, resulting in a tensor of shape (1,) which contains the index of the next token to be generated.
        generated_tokens.append(next_token) 

        if next_token.item() == stop_token:
            break

        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)


def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Performs Nucleus (Top-P) sampling on the predicted probability distribution.

    Args:
        probs (torch.Tensor): The probability distribution for the next token.
        p (float): The cumulative probability threshold (0.0 to 1.0).

    Returns:
        torch.Tensor: The index of the sampled token.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    """
    Main entry point to initialize the environment and trigger model inference.

    Handles device detection (CUDA/MPS/CPU), model weight loading, and 
    configuration-based processor setup.
    """
    device = "cpu"
    print(f"Model path: {model_path}")

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)
