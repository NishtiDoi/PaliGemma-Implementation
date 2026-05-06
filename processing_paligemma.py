from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """Prepends image tokens and a BOS(Beginning-of-Sequence) token to the text prompt.

    Constructs the full input string expected by PaliGemma by prepending a fixed
    number of image placeholder tokens, followed by a BOS token, the text prompt,
    and a trailing newline. The newline is required because the model was trained
    with it as part of every prompt.

    Reference:
        https://huggingface.co/blog/paligemma#detailed-inference-process

    Args:
        prefix_prompt (str): The raw text prompt to be tokenized.
        bos_token (str): The beginning-of-sequence token string (e.g. "<bos>").
        image_seq_len (int): Number of image tokens to prepend, corresponding to
            the number of image patch embeddings produced by the vision encoder.
        image_token (str): The image placeholder token string (e.g. "<image>").

    Returns:
        str: The formatted prompt string in the form:
            ``"<image>" * image_seq_len + bos_token + prefix_prompt + "\\n"``
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
    # when you see <image> * image_seq_len, you are basically performing Sequence Reservation.


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Rescales pixel values of an image array by a scalar factor.

    Typically used to convert uint8 pixel values in [0, 255] to float values
    in [0, 1] by passing ``scale=1/255.0``.

    Args:
        image (np.ndarray): Input image array of any shape.
        scale (float): Multiplicative scale factor applied to every element.
        dtype (np.dtype, optional): Output array dtype. Defaults to
            ``np.float32``.

    Returns:
        np.ndarray: Rescaled image array cast to ``dtype``.
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """Resizes a PIL image to the specified (height, width).

    Args:
        image (PIL.Image.Image): The input PIL image to resize.
        size (Tuple[int, int]): Target size as ``(height, width)``. Note that
            PIL's ``resize`` expects ``(width, height)``, so the values are
            swapped internally.
        resample (PIL.Image.Resampling, optional): Resampling filter to use
            (e.g. ``Image.Resampling.BICUBIC``). Defaults to ``None``, which
            uses PIL's default filter.
        reducing_gap (int, optional): Optimization parameter passed to PIL to
            speed up downsampling by first reducing the image by an integer
            factor. Defaults to ``None``.

    Returns:
        PIL.Image.Image: The resized PIL image.
    """
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """Normalizes an image array by subtracting the mean and dividing by std.

    Performs per-channel normalization: ``(image - mean) / std``. Commonly
    applied after rescaling to [0, 1] to zero-center and unit-scale each
    channel, which stabilizes training and inference for many vision models.

    Args:
        image (np.ndarray): Input image array, typically of shape
            ``(H, W, C)`` and dtype ``float32``.
        mean (float or Iterable[float]): Per-channel mean values. If a scalar,
            the same value is applied to all channels.
        std (float or Iterable[float]): Per-channel standard deviation values.
            If a scalar, the same value is applied to all channels.

    Returns:
        np.ndarray: Normalized image array of the same shape and dtype as the
        input.
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """Applies the full preprocessing pipeline to a list of PIL images.

    Sequentially performs:
      1. **Resize** — each image is resized to ``(size[0], size[1])``.
      2. **Convert to array** — each PIL image is converted to a NumPy array.
      3. **Rescale** — pixel values are multiplied by ``rescale_factor`` to
         bring them into [0, 1].
      4. **Normalize** — per-channel mean subtraction and std division.
      5. **Transpose** — axes are reordered from ``(H, W, C)`` to
         ``(C, H, W)`` to match PyTorch's expected tensor layout.

    Args:
        images (List[PIL.Image.Image]): List of input PIL images to process.
        size (Dict[str, int], optional): A sequence whose first two elements
            are ``(height, width)`` for resizing. Defaults to ``None``.
        resample (PIL.Image.Resampling, optional): Resampling filter used
            during resizing. Defaults to ``None``.
        rescale_factor (float, optional): Scalar multiplied against raw pixel
            values, e.g. ``1/255.0``. Defaults to ``None``.
        image_mean (float or List[float], optional): Per-channel mean used for
            normalization. Defaults to ``None``.
        image_std (float or List[float], optional): Per-channel standard
            deviation used for normalization. Defaults to ``None``.

    Returns:
        List[np.ndarray]: List of preprocessed image arrays, each with shape
        ``(C, H, W)`` and dtype ``float32``.
    """
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:
    """Processor for PaliGemma vision-language model inputs.

    Handles joint preprocessing of images and text prompts into the tensor
    format expected by PaliGemma. Images are resized, rescaled, and normalized
    before being stacked into a batch tensor. Text prompts are prefixed with
    the required image placeholder tokens and then tokenized.

    The tokenizer is extended with a special ``<image>`` token, 1024 location
    tokens (``<loc0000>`` … ``<loc1023>``) for bounding-box tasks, and 128
    segmentation tokens (``<seg000>`` … ``<seg127>``).

    Reference:
        https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer

    Attributes:
        IMAGE_TOKEN (str): The image placeholder token string ``"<image>"``.
        image_seq_length (int): Number of image tokens prepended to each
            prompt, equal to the number of patch embeddings from the vision
            encoder.
        image_size (int): Height and width (in pixels) that input images are
            resized to before being passed to the model.
        image_token_id (int): Integer token ID corresponding to ``IMAGE_TOKEN``
            in the vocabulary.
        tokenizer: The extended HuggingFace tokenizer instance.
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """Initializes the PaliGemmaProcessor.

        Extends the provided tokenizer with the image placeholder token, object
        detection location tokens, and segmentation tokens. 
        Also disables automatic BOS/EOS insertion since both are managed manually.

        Args:
            tokenizer: A HuggingFace ``PreTrainedTokenizer`` (or compatible)
                instance. Will be mutated in-place to add special tokens.
            num_image_tokens (int): Number of ``<image>`` tokens to prepend to
                every prompt. Should match the number of visual patch embeddings
                produced by the vision encoder (e.g. 256 for a 224×224 image
                with 14×14 patches).
            image_size (int): Square resolution (height == width) in pixels
                that all input images are resized to.
        """

        super().__init__() # calls parent class constructor

        self.image_seq_length = num_image_tokens # config values as instance attribute
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]} # adds the <image> token to the tokenizer's vocabulary as a special token, which does not get tokenized
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [  # These tokens are used for object detection (bounding boxes)
            f"<loc{i:04d}>" for i in range(1024)
        ] 
        EXTRA_TOKENS += [ # These tokens are used for object segmentation
            f"<seg{i:03d}>" for i in range(128)
        ]  
        tokenizer.add_tokens(EXTRA_TOKENS) # adds the location and segmentation tokens to the tokenizer's vocabulary as regular tokens, which do get tokenized
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN) #Looks up the integer ID assigned to <image> after it was added. This ID is used later when the model needs to know which positions in the input sequence correspond to image patches (so it can replace them with vision encoder embeddings).
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__( # when you call the processor instance, this method gets executed, runs two parallel pipelines and merges them
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """Preprocesses a single image–prompt pair into model-ready tensors.

        Applies the full image preprocessing pipeline (resize → rescale →
        normalize → channel-first transpose) and constructs the tokenized text
        input with the required image token prefix.

        Note:
            Currently only supports a batch size of 1; exactly one image and
            one prompt must be provided.

        Args:
            text (List[str]): A list containing a single text prompt string.
            images (List[PIL.Image.Image]): A list containing a single PIL
                image.
            padding (str, optional): Padding strategy passed to the tokenizer
                (e.g. ``"longest"`` or ``"max_length"``). Defaults to
                ``"longest"``.
            truncation (bool, optional): Whether to truncate sequences that
                exceed the tokenizer's maximum length. Defaults to ``True``.

        Returns:
            dict: A dictionary with the following keys:

            - **pixel_values** (``torch.Tensor``): Preprocessed image tensor
              of shape ``(1, C, H, W)`` with dtype ``float32``.
            - **input_ids** (``torch.Tensor``): Token ID tensor of shape
              ``(1, seq_len)``.
            - **attention_mask** (``torch.Tensor``): Attention mask tensor of
              shape ``(1, seq_len)``.

        Raises:
            AssertionError: If ``len(images) != 1`` or ``len(text) != 1``.
        """
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # IMAGE PIPELINE
        pixel_values = process_images(# Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # TEXT PIPELINE
        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,       # ["<image>...<image><bos>What is in this image?\n"]
            return_tensors="pt", # give me PyTorch tensors, not lists
            padding=padding,     # how to handle different length sequences
            truncation=truncation
        )
       # "<image><image><image>...<image><bos>What is in this image?\n"
       #|_______ 256 tokens _______|     |_______ text tokens ________|
# The tokenizer processes this string and produces two tensors: 1. input_ids 2. attention_mask. 
# The input_ids tensor contains the integer token IDs corresponding to each token in the input string, including the image tokens and text tokens. 
# The attention_mask tensor contains 1s for positions that correspond to real tokens (both image and text) and 0s for any padding tokens (if padding was applied).

        return_data = {"pixel_values": pixel_values, **inputs} 
        # inputs is already a dict {"input_ids": ..., "attention_mask": ...}. The **inputs unpacks it, and the whole line creates a new dict that combines the pixel_values with the tokenized inputs, resulting in a dict with keys "pixel_values", "input_ids", and "attention_mask".
        return return_data
