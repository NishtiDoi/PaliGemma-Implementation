from typing import Optional, Tuple
import torch
import torch.nn as nn

# This file implements the Vision Encoder for the SigLIP (Sigmoid Language-Image Pre-training) model. Its primary purpose is to take raw image pixels and transform them into a sequence of meaningful mathematical vectors (embeddings) that an AI model can "read."

class SiglipVisionConfig: # Stores hyperparams

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):  # This class answers the question: how do you turn a 2D image into something a transformer can process?
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=3,     # RGB image
            out_channels=768,  # one 768-dim vector per patch
            kernel_size=16,    # look at 16×16 pixel blocks
            stride=16,         # jump 16 pixels each time (no overlap)
            padding="valid"    # don't add any border pixels
        ) #196 patches total. This is why the processor prepended 196 <image> tokens — one per patch.

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        # Why is this needed? After the Conv2d, the model has 196 patch vectors but no idea where each patch came from spatially. 
        # Patch 50 and patch 150 look the same to the transformer without this. Position embeddings inject location information.
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # Just creates a fixed tensor [[0, 1, 2, ..., 195]] — the indices used to look up position embeddings. register_buffer means it moves to GPU automatically with the model but is not a learned parameter (no gradients). persistent=False means it's not saved in the model checkpoint.
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Input: (B, C, H, W) → batch of images
        _, _, height, width = pixel_values.shape  

        # Step 1: Split image into non-overlapping patches and project each patch
        # Output: (B, embed_dim, H/patch, W/patch)
        patch_embeds = self.patch_embedding(pixel_values)

        # Step 2: Flatten spatial grid into a sequence of patches
        # (B, embed_dim, H_p, W_p) → (B, embed_dim, num_patches)
        embeddings = patch_embeds.flatten(2)

        # Step 3: Rearrange to match transformer input format
        # (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        embeddings = embeddings.transpose(1, 2)

        # Step 4: Add positional information so model knows patch locations
        # position_embedding: (1, num_patches, embed_dim)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # Output: (B, num_patches, embed_dim) → sequence of image tokens
        return embeddings


class SiglipAttention(nn.Module): # How patches interact globally 
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size # 768
        self.num_heads = config.num_attention_heads # 12 parallel attention heads, each with its own set of QKV projections. This allows the model to attend to different aspects of the input simultaneously.
        self.head_dim = self.embed_dim // self.num_heads #768 / 12 = 64. Each attention head operates in a 64-dimensional subspace of the full 768-dimensional embedding space.
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

# Projection layers
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    """
    forward — what’s happening (compressed view)
    Start:
    Input is patch embeddings [B, N, 768]
    Each patch is independent at this point.
    Project to Q, K, V:
    Create three versions of each patch for matching and information flow.
    Split into heads:
    Break 768 → multiple smaller chunks so attention can run in parallel.
    Compute attention (Q · Kᵀ):
    Each patch scores how relevant every other patch is.
    Softmax:
    Turn scores into probabilities (who to focus on).
    Weighted sum with V:
    Each patch gathers information from others based on attention.
    Merge heads:
    Combine all parallel attention results back into one vector per patch.
    Final projection:
    Mix everything and return updated patch embeddings.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # You now have 12 parallel attention streams, each seeing the full sequence.

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module): # How each patch processes its own information after attending to other patches. This is where the model learns to transform the attended information into more complex features. For example, it might learn to combine the information from the "wheel" and "door" patches to create a new feature that represents "car."
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Expands the representation to a higher dimension to extract complex features
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Shrinks the representation back to the original embedding size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Step 1: Up-projection (Expansion)
        # Increase dimensionality to give the model "room" to compute complex patterns
        hidden_states = self.fc1(hidden_states)

        # Step 2: Activation Function (Non-linearity)
        # GELU allows the model to learn non-linear relationships; "tanh" makes it faster to compute
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        # Step 3: Down-projection (Contraction)
        # Reduce back to original size so it can be added to the residual connection later
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module): # one transformer layer, repeated 12 times in the encoder. Each layer has its own attention and MLP modules, and its own set of parameters.
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)  # lets patches talk to each other globally, across the whole image. This is where the model learns relationships between different parts of the image, like "the patch with the wheel is near the patch with the door, so this is probably a car."
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config) # processes each patch independently after the attention step. This is where the model learns to transform the attended information into more complex features. For example, it might learn to combine the information from the "wheel" and "door" patches to create a new feature that represents "car."
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Processes data through 2 main sub-blocks: 1. Self-Attention and 2. MLP, with residual connections and layer normalization around each.
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Save input for the skip-connection (addition)
        residual = hidden_states
        
        # Normalize and run Attention (Global context)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        
        # Add the original input back (Residual Connection)
        hidden_states = residual + hidden_states
        
        # Save state again for the next skip-connection
        residual = hidden_states
        
        # Normalize and run MLP (Local refinement per patch)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add the state from the previous step back
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipEncoder(nn.Module): # the 12 blocks stacked 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module): # Full pipreling wired together: from raw pixels to final patch embeddings that get fed into the language model. This is the "vision encoder" part of the overall Siglip architecture.
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module): # Wraps the vision transformer and provides a clean interface for the rest of the architecture to call. The language model doesn't need to know anything about how images are processed, it just gets a sequence of embeddings that represent the image.

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 
