import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modelling_siglip import SiglipVisionConfig, SiglipVisionModel

# This code defines the complete architecture for PaliGemma, a Vision-Language Model (VLM). 
# It connects a Vision Transformer (SigLIP) to a Large Language Model (Gemma) so the model can "see" images and talk about them.

class KVCache(): # Stores Key and Value tensors from previous time steps during text generation so the model doesn't have to recompute them, making inference much faster.

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig(): # Stores hyperparameters for the Gemma language model

    def __init__(
        self,
        vocab_size,
        hidden_size, # embedding dimension 
        intermediate_size, # MLP expansion which is usually 4x hidden
        num_hidden_layers, # number of decoder blocks stacked 
        num_attention_heads, # total Q heads
        num_key_value_heads, # number of K/V heads (if using GQA, otherwise same as num_attention_heads)
        head_dim=256, # size per dim
        max_position_embeddings=8192, # max context length 
        
        # training ke liye 
        rms_norm_eps=1e-6, 
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig(): # Stores hyperparameters for the entire PaliGemma model, including both the vision and language components. (VLM ka config)

    def __init__(
        self,
        vision_config=None, # passed to the SiglipVisionModel which processes images
        text_config=None, # passed to the GemmaConfig which processes text and generates output
        ignore_index=-100, # index to ignore when computing the loss
        image_token_index=256000, # token index for image tokens
        vocab_size=257152, # size of the vocabulary
        
        projection_dim=2048, # dimension of the projection layer ALSO SigLIP output ≠ Gemma input by default.
        hidden_size=2048, # size of the hidden layers
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module): # a normalisation layer used to stabilise the hidden states, ensuring values dont explode as they pass
    #Standard LayerNorm centers the data by subtracting the mean ($\mu$) and then scales it. RMSNorm skips the centering step, which reduces computational overhead by about 40% while providing similar stabilization benefits for deep networks.
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # A very small constant ($1 x 10^-6 added to the denominator to prevent division by zero.
        self.weight = nn.Parameter(torch.zeros(dim)) # A learnable parameter (scaling factor) initialized to zeros. In Gemma's specific implementation, this weight is applied as $(1 + \text{weight})$, which is why it starts at zero (making the initial multiplier $1.0$).

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # Calculates the mean of the squares of the hidden states along the last dimension.
                                                                           # Then it takes the reciprocal of the square root of this mean (plus epsilon for stability) to get the normalization factor. Finally, it multiplies the input tensor x by this normalization factor to produce the normalized output. 

    def forward(self, x):
        output = self._norm(x.float()) # The input is cast to float32 for precision during the calculation of squares and square roots, avoiding numerical instability.
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module): # RoPE (Rotary Positional Embedding), injects positional information by rotating the Query and Key vectors, helping the model understand the relative order of tokens.
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(-2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaMLP(nn.Module): # A feed-forward network that processes each token individually to refine its features.
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # gate_proj: Projects input to a higher dimension to act as a "filter" or "gate"
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        
        # up_proj: Parallel projection to a higher dimension that carries the "content"
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        
        # down_proj: Projects the merged high-dim representation back to the original model dimension
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # 1. gate_proj(x): Linear expansion
        # 2. nn.functional.gelu(..., approximate="tanh"): Applies GELU activation to the gate
        # 3. ... * self.up_proj(x): Element-wise multiplication (the "Gated Linear Unit" mechanism)
        # 4. self.down_proj(...): Linear contraction back to hidden_size
        
        # This implementation uses the GeGLU activation variant common in modern LLMs
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Broadcasts KV heads to match the number of Query heads.
    Example: If you have 8 KV heads and 32 Query heads, n_rep would be 4.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # If the number of KV heads already matches Query heads, do nothing.
    if n_rep == 1:
        return hidden_states
    
    # 1. Add a 'singleton' dimension at index 2 (None) to act as a placeholder for repetition.
    # New Shape: [batch, num_key_value_heads, 1, slen, head_dim]
    hidden_states = hidden_states[:, :, None, :, :]
    
    # 2. Use .expand() to virtually repeat the KV heads n_rep times.
    # This is memory efficient because it doesn't allocate new memory for copies.
    # New Shape: [batch, num_key_value_heads, n_rep, slen, head_dim]
    hidden_states = hidden_states.expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # 3. Collapse the num_key_value_heads and n_rep dimensions together.
    # Final Shape: [batch, num_key_value_heads * n_rep, slen, head_dim]
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module): #Multi-head grouped query attention (GQA) with rotary position embeddings (RoPE).
    """
    This is the core attention mechanism for the Gemma language model. It differs
    from standard multi-head attention (as used in SigLIP) in two important ways:

    1. **Grouped Query Attention (GQA):** Instead of one K/V head per Q head,
       multiple Q heads share a single K/V head. This significantly reduces
       memory usage and speeds up inference, especially with a KV cache.

       Example with typical Gemma config:
           num_attention_heads (Q)  = 8   (8 query heads)
           num_key_value_heads (KV) = 1   (1 key/value head shared by all)
           num_key_value_groups     = 8   (each KV head serves 8 Q heads)

    2. **Rotary Position Embeddings (RoPE):** Instead of adding position info
       to the input (like SigLIP's position embeddings), RoPE rotates the Q
       and K vectors based on their position. This lets the model generalise
       to sequence lengths it wasn't explicitly trained on.

    Attributes:
        layer_idx (int): Index of this layer in the decoder stack. Used to
            read/write the correct slot in the KV cache.
        num_heads (int): Number of query attention heads.
        num_key_value_heads (int): Number of key/value heads (fewer than Q heads in GQA).
        num_key_value_groups (int): How many Q heads share each KV head.
            Computed as ``num_heads // num_key_value_heads``.
        head_dim (int): Dimensionality of each attention head.
        is_causal (bool): Always True — Gemma is a causal (left-to-right) decoder.
        q_proj: Projects hidden states to query vectors (full num_heads size).
        k_proj: Projects hidden states to key vectors (smaller num_key_value_heads size).
        v_proj: Projects hidden states to value vectors (smaller num_key_value_heads size).
        o_proj: Projects concatenated head outputs back to hidden_size.
        rotary_emb: RoPE module that computes cos/sin rotation coefficients.
    """

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        """Initialises GemmaAttention with projections and RoPE.

        Args:
            config (GemmaConfig): Model hyperparameters. Key fields used:
                - ``hidden_size``: Input/output feature dimension.
                - ``num_attention_heads``: Number of Q heads.
                - ``num_key_value_heads``: Number of K/V heads (GQA).
                - ``head_dim``: Per-head feature dimension.
                - ``rope_theta``: Base frequency for RoPE (default 10000).
                - ``attention_bias``: Whether linear projections use bias.
            layer_idx (int, optional): Position of this layer in the decoder
                stack. Required when using a KV cache so each layer writes
                to the correct cache slot.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        # How many Q heads share each single KV head.
        # e.g. 8 Q heads / 1 KV head = 8 groups
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True  # Gemma is a causal decoder — no peeking at future tokens

        # Sanity check: hidden_size must divide evenly across heads
        assert self.hidden_size % self.num_heads == 0

        # --- Projection layers ---
        # Q gets the full num_heads allocation
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # K and V only get num_key_value_heads — this is the GQA saving
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # Output projection merges all heads back to hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # RoPE: generates cos/sin rotation coefficients per position
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Computes grouped query attention with RoPE and optional KV caching.

        The forward pass has six logical stages:
          1. Project hidden states into Q, K, V vectors.
          2. Reshape into per-head format.
          3. Apply RoPE to Q and K (inject position information).
          4. Update and retrieve the KV cache (if provided).
          5. Expand K/V heads to match Q head count (GQA).
          6. Compute scaled dot-product attention and project output.

        Args:
            hidden_states (torch.Tensor): Input token representations of shape
                ``(batch_size, seq_len, hidden_size)``. Contains both image
                patch embeddings and text token embeddings interleaved.
            attention_mask (torch.Tensor, optional): Additive mask of shape
                ``(batch_size, num_heads, seq_len_q, seq_len_kv)``.
                ``0`` means attend, ``-inf`` means block. Applied before
                softmax so masked positions get ~0 probability.
            position_ids (torch.LongTensor, optional): Token positions of shape
                ``(batch_size, seq_len)``. Used by RoPE to compute the correct
                rotation angle for each position.
            kv_cache (KVCache, optional): Cache of past K/V tensors for fast
                autoregressive generation. If provided, new K/V states are
                appended to the cache and the full history is returned.
            **kwargs: Absorbed for API compatibility.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - **attn_output** ``(batch_size, seq_len, hidden_size)``:
                  Attention output projected back to hidden size.
                - **attn_weights** ``(batch_size, num_heads, seq_len_q, seq_len_kv)``:
                  Softmaxed attention scores (useful for visualisation/debugging).
        """

        bsz, q_len, _ = hidden_states.size()  # [Batch_Size, Seq_Len, Hidden_Size]

        # ------------------------------------------------------------------ #
        # STAGE 1 — Project to Q, K, V                                       #
        # ------------------------------------------------------------------ #
        # Q gets projected to the full num_heads size
        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)

        # K and V are projected to the smaller num_key_value_heads size (GQA)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # ------------------------------------------------------------------ #
        # STAGE 2 — Reshape to per-head format                               #
        # ------------------------------------------------------------------ #
        # Split the last dimension into (num_heads, head_dim) then move
        # num_heads to dimension 1 so each head can be processed in parallel.

        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        #   -> [Batch_Size, Seq_Len, Num_Heads_Q, Head_Dim]
        #   -> [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Same reshape for K and V but with num_key_value_heads (smaller)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ------------------------------------------------------------------ #
        # STAGE 3 — Apply RoPE to inject position information                #
        # ------------------------------------------------------------------ #
        # RoPE computes a rotation matrix for each position.
        # cos and sin: [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        # Rotate Q and K vectors by their respective position angles.
        # K is rotated too (not just Q) so relative position is encoded in Q·K^T.
        # Shapes unchanged: Q still [Batch, Num_Heads_Q, Seq_Len, Head_Dim]
        #                   K still [Batch, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ------------------------------------------------------------------ #
        # STAGE 4 — KV Cache update                                          #
        # ------------------------------------------------------------------ #
        # During generation, instead of recomputing K/V for all past tokens,
        # we append the new token's K/V to the cache and retrieve the full
        # history. This is the key to fast autoregressive inference.
        #
        # Prefill (first pass):  cache is empty → just stores K/V
        # Decode (subsequent):   new K/V appended → returns full history
        if kv_cache is not None:
            # key_states and value_states now contain ALL tokens (past + current)
            # shape: [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim]
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # ------------------------------------------------------------------ #
        # STAGE 5 — Expand KV heads to match Q heads (GQA)                  #
        # ------------------------------------------------------------------ #
        # Q has num_heads heads, but K/V only have num_key_value_heads.
        # repeat_kv tiles each KV head num_key_value_groups times so the
        # shapes are compatible for the matmul.
        #
        # Example: 1 KV head, 8 Q heads → KV head is repeated 8 times
        # [Batch, Num_Heads_KV, Seq_Len, Head_Dim]
        #   -> [Batch, Num_Heads_Q, Seq_Len, Head_Dim]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ------------------------------------------------------------------ #
        # STAGE 6 — Scaled dot-product attention                             #
        # ------------------------------------------------------------------ #

        # Q · K^T / sqrt(head_dim)
        # [Batch, Num_Heads_Q, Seq_Len_Q, Head_Dim] x [Batch, Num_Heads_Q, Head_Dim, Seq_Len_KV]
        # -> [Batch, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        # Each row is a query token's raw attention score against every key token.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        # Add the mask (0 = attend, -inf = block). Positions with -inf become
        # ~0 after softmax, effectively preventing attention to those tokens.
        attn_weights = attn_weights + attention_mask

        # Softmax across the key dimension (dim=-1) — converts raw scores to
        # probabilities. Done in float32 for numerical stability, then cast back.
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Dropout applied to attention weights during training only
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Weighted sum of value vectors using attention probabilities.
        # [Batch, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV] x [Batch, Num_Heads_Q, Seq_Len_KV, Head_Dim]
        # -> [Batch, Num_Heads_Q, Seq_Len_Q, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Move heads back to 3rd dimension before merging
        # [Batch, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Merge all heads into a single vector per token position
        # [Batch, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)

        # Final linear projection to mix information across heads
        # [Batch, Seq_Len_Q, Num_Heads_Q * Head_Dim] -> [Batch, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):# A single Transformer block combining Attention and MLP.

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module): # The main model that orchestrates the vision and language components.

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    # Ignore copy
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states

class GemmaForCausalLM(nn.Module): #Gemma language model with a causal language modelling head.
    """
    "Causal LM" means the model predicts the next token given all previous
    tokens — it cannot look ahead. This is how all autoregressive text
    generation works: GPT, LLaMA, Gemma, etc.

    This class is a thin wrapper that adds one extra layer on top of
    GemmaModel: a linear projection called lm_head that converts the
    model's internal hidden vectors into a probability distribution over
    the entire vocabulary.

    Architecture:
        inputs_embeds (merged image + text vectors)
               │
               ▼
          GemmaModel                  ← N decoder layers (attention + MLP)
               │
               ▼  [Batch, Seq_Len, Hidden_Size]
            lm_head                   ← Linear(hidden_size → vocab_size)
               │
               ▼  [Batch, Seq_Len, Vocab_Size]
             logits                   ← raw score for every token at every position

    The logit at position i represents "given tokens 0..i, how likely is
    each vocabulary token to come next?" The highest scoring token is
    typically chosen as the next generated token.

    Weight tying:
        lm_head.weight is shared with the input token embedding matrix
        (embed_tokens). This is a standard technique — the same matrix
        that maps token IDs → vectors on the way in also maps hidden
        vectors → token scores on the way out. It reduces parameters
        and often improves performance.

    Attributes:
        model (GemmaModel): The core transformer decoder stack.
        vocab_size (int): Number of tokens in the vocabulary.
        lm_head (nn.Linear): Projects hidden states to vocabulary logits.
            No bias, and its weight is tied to embed_tokens.
    """

    def __init__(self, config):
        """Initialises the model, decoder stack, and lm_head projection.

        Args:
            config (GemmaConfig): Hyperparameters. Key fields:
                - ``hidden_size``: Dimension of internal token representations.
                - ``vocab_size``: Number of tokens in the vocabulary.
                  lm_head output size matches this exactly.
        """
        super().__init__()
        self.config = config

        # The full transformer decoder stack (embeddings + N decoder layers + norm)
        self.model = GemmaModel(config)

        self.vocab_size = config.vocab_size

        # The language modelling head: converts each hidden vector into a
        # score for every token in the vocabulary.
        # hidden_size → vocab_size, no bias (standard for LM heads)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        """Returns the token embedding table from the underlying GemmaModel.

        Used by PaliGemmaForConditionalGeneration to convert input_ids into
        embedding vectors before merging with image features.

        Returns:
            nn.Embedding: The embed_tokens layer of shape
            ``(vocab_size, hidden_size)``.
        """
        return self.model.embed_tokens

    def tie_weights(self):
        """Ties lm_head weights to the input token embedding matrix.

        Weight tying means lm_head.weight and embed_tokens.weight point to
        the exact same tensor in memory. The intuition:

            Input side:  token ID  → embedding vector  (embed_tokens)
            Output side: hidden vector → token scores  (lm_head)

        Both operations are conceptually inverses of each other, so sharing
        the matrix works well in practice and halves the parameter count for
        this large matrix (vocab_size × hidden_size).

        After calling this, updating one automatically updates the other.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """Runs the decoder stack and projects outputs to vocabulary logits.

        Note that this method receives inputs_embeds (already-embedded
        vectors), NOT raw input_ids. The embedding step and image/text
        merging happen upstream in PaliGemmaForConditionalGeneration before
        this is called.

        Args:
            attention_mask (torch.Tensor, optional): Additive causal mask of
                shape ``(batch_size, num_heads, seq_len_q, seq_len_kv)``.
                Built by _merge_input_ids_with_image_features upstream.
            position_ids (torch.LongTensor, optional): Token positions of
                shape ``(batch_size, seq_len)``. Used by RoPE inside each
                attention layer.
            inputs_embeds (torch.FloatTensor, optional): Pre-computed token
                embeddings of shape ``(batch_size, seq_len, hidden_size)``.
                Contains the merged image patch vectors and text token vectors.
            kv_cache (KVCache, optional): Cache of past K/V tensors. If
                provided, only the new token needs to be processed on each
                generation step rather than the full sequence.

        Returns:
            dict: Always contains:
                - **logits** ``(batch_size, seq_len, vocab_size)`` as float32:
                  Raw unnormalised scores for every vocabulary token at every
                  sequence position. To get the next token, take
                  ``logits[:, -1, :].argmax(-1)``.

            If ``kv_cache`` is not None, also contains:
                - **kv_cache** (KVCache): The updated cache with the current
                  step's K/V tensors appended. Passed back to the caller so
                  it can be reused on the next generation step.
        """

        # Run the full transformer decoder stack.
        # inputs_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs:       [Batch_Size, Seq_Len, Hidden_Size]
        # Each position's hidden vector now encodes information from all
        # previous positions (via causal attention) and itself.
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs  # [Batch_Size, Seq_Len, Hidden_Size]

        # Project each hidden vector to a score for every vocabulary token.
        # This is the "what comes next?" question asked at every position.
        # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Vocab_Size]
        logits = self.lm_head(hidden_states)

        # Cast to float32 for numerical stability during sampling/argmax,
        # even if the model ran in bfloat16 or float16.
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated KV cache so the caller can reuse it on the
            # next generation step without reprocessing past tokens.
            return_data["kv_cache"] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
    
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # Make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
