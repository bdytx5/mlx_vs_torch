# # Copyright © 2023 Apple Inc.

# import argparse
# import time
# from dataclasses import dataclass
# from typing import Optional, Tuple

# import mlx.core as mx
# import mlx.nn as nn
# from mlx.utils import tree_unflatten
# from tiktoken import get_encoding  # Replace SentencePiece with tiktoken






# import time

# def tic():
#     """Start a timer and return the current time."""
#     return time.time()

# def toc(msg, start):
#     """Calculate the elapsed time and print a formatted message."""
#     end = time.time()
#     return f"[INFO] {msg}: {end - start:.3f} s"




# class LayerNorm(nn.Module):
#     r"""Applies layer normalization [1] on the inputs.

#     Computes

#     .. math::

#         y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

#     where :math:`\gamma` and :math:`\beta` are learned per feature dimension
#     parameters initialized at 1 and 0 respectively.

#     [1]: https://arxiv.org/abs/1607.06450

#     Args:
#         dims (int): The feature dimension of the input to normalize over
#         eps (float): A small additive constant for numerical stability
#         affine (bool): If True learn an affine transform to apply after the
#             normalization
#     """

#     def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = False):
#         super().__init__()
#         if affine:
#             self.bias = bias
#             if bias:
#                 self.bias = mx.zeros((dims,))
#             self.weight = mx.ones((dims,))
#         self.eps = eps
#         self.dims = dims

#     def _extra_repr(self):
#         return f"{self.dims}, eps={self.eps}, affine={'weight' in self}, bias={self.bias}"

#     def __call__(self, x):
#         means = mx.mean(x, axis=-1, keepdims=True)
#         var = mx.var(x, axis=-1, keepdims=True)
#         x = (x - means) * mx.rsqrt(var + self.eps)
#         if self.bias:
#             return (self.weight * x + self.bias) if "weight" in self else x
#         else:
#             return (self.weight * x) if "weight" in self else x


# @dataclass
# class ModelArgs:
#     dim: int
#     n_layers: int
#     head_dim: int
#     hidden_dim: int
#     n_heads: int
#     n_kv_heads: int
#     norm_eps: float
#     vocab_size: int
#     rope_theta: float
#     rope_traditional: bool = True


# class Attention(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.args = args

#         self.n_heads: int = args.n_heads
#         self.n_kv_heads: int = args.n_kv_heads

#         self.repeats = self.n_heads // self.n_kv_heads

#         self.scale = self.args.head_dim**-0.5

#         self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
#         self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
#         self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
#         self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
#         self.rope = nn.RoPE(
#             args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
#         )

#     def __call__(
#         self,
#         x: mx.array,
#         mask: Optional[mx.array] = None,
#         cache: Optional[Tuple[mx.array, mx.array]] = None,
#     ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
#         B, L, D = x.shape

#         queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

#         queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
#         keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
#         values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

#         def repeat(a):
#             a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
#             return a.reshape([B, self.n_heads, L, -1])

#         keys, values = map(repeat, (keys, values))

#         if cache is not None:
#             key_cache, value_cache = cache
#             queries = self.rope(queries, offset=key_cache.shape[2])
#             keys = self.rope(keys, offset=key_cache.shape[2])
#             keys = mx.concatenate([key_cache, keys], axis=2)
#             values = mx.concatenate([value_cache, values], axis=2)
#         else:
#             queries = self.rope(queries)
#             keys = self.rope(keys)

#         scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
#         if mask is not None:
#             scores += mask
#         scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
#         output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
#         return self.wo(output), (keys, values)


# class FeedForward(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()

#         self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
#         self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
#         self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

#     def __call__(self, x) -> mx.array:
#         return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# class TransformerBlock(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.n_heads = args.n_heads
#         self.dim = args.dim
#         self.attention = Attention(args)
#         self.feed_forward = FeedForward(args=args)
#         self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
#         self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
#         self.args = args

#     def __call__(
#         self,
#         x: mx.array,
#         mask: Optional[mx.array] = None,
#         cache: Optional[Tuple[mx.array, mx.array]] = None,
#     ) -> mx.array:
#         r, cache = self.attention(self.attention_norm(x), mask, cache)
#         h = x + r
#         r = self.feed_forward(self.ffn_norm(h))
#         out = h + r
#         return out, cache


# class Llama(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.args = args
#         self.vocab_size = args.vocab_size
#         self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
#         self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
#         self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
#         self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

#     def __call__(self, x):
#         mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
#         mask = mask.astype(self.tok_embeddings.weight.dtype)

#         x = self.tok_embeddings(x)
#         for l in self.layers:
#             x, _ = l(x, mask)
#         x = self.norm(x)
#         return self.output(x)

#     def generate(self, x, temp=1.0):
#         def sample(logits):
#             if temp == 0:
#                 return mx.argmax(logits, axis=-1)
#             else:
#                 return mx.random.categorical(logits * (1 / temp))

#         cache = []
#         mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
#         mask = mask.astype(self.tok_embeddings.weight.dtype)

#         x = self.tok_embeddings(x)
#         for l in self.layers:
#             x, c = l(x, mask=mask)
#             cache.append(c)
#         x = self.norm(x)
#         y = self.output(x[:, -1])
#         y = sample(y)
#         yield y

#         while True:
#             x = y[:, None]

#             x = self.tok_embeddings(x)
#             for i in range(len(cache)):
#                 x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
#             x = self.norm(x)
#             y = sample(self.output(x[:, -1]))

#             yield y


# def load_model(model_path):
#     # Hardcoded ModelArgs for demonstration purposes
#     # config = {
#     #     "dim": 512,
#     #     "n_layers": 6,
#     #     "head_dim": 64,
#     #     "hidden_dim": 2048,
#     #     "n_heads": 8,
#     #     "n_kv_heads": 8,
#     #     "norm_eps": 1e-5,
#     #     "vocab_size": 50257,  # Typical GPT-2 vocab size
#     #     "rope_theta": 10000,
#     # }
# # Updated configuration to match 'gpt2' (124M parameters)
#     config = {
#         "dim": 768,  # Embedding dimension size (matches n_embd in GPT-2)
#         "n_layers": 12,  # Number of transformer layers
#         "head_dim": 64,  # Size of each attention head's dimension
#         "hidden_dim": 3072,  # Typically, hidden_dim is 4 * n_embd
#         "n_heads": 12,  # Number of attention heads
#         "n_kv_heads": 12,  # Number of key-value heads (matches n_heads)
#         "norm_eps": 1e-5,  # Epsilon for normalization layers
#         "vocab_size": 50257,  # Standard GPT-2 vocabulary size
#         "rope_theta": 10000,  # Default RoPE parameter
#     }

#     model = Llama(ModelArgs(**config))

#     # Use GPT-2 Tokenizer
#     tokenizer = get_encoding("gpt2")

#     return model, tokenizer


# def generate(args):
#     input("Press enter to start generation")
#     print("------")
#     print(args.prompt)

#     # Encode the prompt using the GPT-2 tokenizer
#     start_ids = tokenizer.encode(args.prompt, allowed_special={"<|endoftext|>"})
#     x = mx.array([start_ids])
#     skip = 0
#     prompt_processing = None
#     tokens = []
#     start = tic()

#     for token in model.generate(x, args.temp):
#         tokens.append(token)

#         if len(tokens) == 1:
#             mx.eval(token)
#             prompt_processing = toc("Prompt processing", start)

#         if len(tokens) >= args.max_tokens:
#             break

#         elif (len(tokens) % args.write_every) == 0:
#             mx.eval(tokens)
#             s = tokenizer.decode([t.item() for t in tokens])
#             print(s[skip:], end="", flush=True)
#             skip = len(s)

#     mx.eval(tokens)
#     full_gen = toc("Full generation", start)
#     s = tokenizer.decode([t.item() for t in tokens])
#     print(s[skip:], flush=True)
#     print("------")
#     print(prompt_processing)
#     print(full_gen)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Llama inference script")
#     parser.add_argument(
#         "--model-path",
#         help="Path to the model weights and tokenizer",
#         default="mlx_model",
#     )
#     parser.add_argument(
#         "--prompt",
#         help="The message to be processed by the model. Ignored when --few-shot is provided.",
#         default="In the beginning the Universe was created.",
#     )
#     parser.add_argument(
#         "--max-tokens", "-m", type=int, default=100, help="How many tokens to generate"
#     )
#     parser.add_argument(
#         "--write-every", type=int, default=1, help="After how many tokens to detokenize"
#     )
#     parser.add_argument(
#         "--temp", type=float, default=0.0, help="The sampling temperature"
#     )
#     parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

#     args = parser.parse_args()

#     mx.random.seed(args.seed)

#     # Load the model and tokenizer
#     model, tokenizer = load_model(args.model_path)

#     generate(args)





# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from base import BaseModelArgs, create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
    n_positions: int
    layer_norm_epsilon: float
    vocab_size: int
    num_key_value_heads: int = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.n_head


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.n_embd % args.n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.head_dim = self.n_embd // self.n_head

        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_embd = args.n_embd
        self.c_fc = nn.Linear(self.n_embd, 4 * self.n_embd)
        self.c_proj = nn.Linear(4 * self.n_embd, self.n_embd)

    def __call__(self, x) -> mx.array:
        return self.c_proj(nn.gelu_approx(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.layer_norm_epsilon = args.layer_norm_epsilon
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.ln_1 = nn.LayerNorm(
            self.n_embd,
            eps=self.layer_norm_epsilon,
        )
        self.ln_2 = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attn(self.ln_1(x), mask, cache)
        h = x + r
        r = self.mlp(self.ln_2(h))
        out = h + r
        return out


class GPT2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_embd = args.n_embd
        self.n_positions = args.n_positions
        self.vocab_size = args.vocab_size
        self.n_layer = args.n_layer
        self.layer_norm_epsilon = args.layer_norm_epsilon
        assert self.vocab_size > 0
        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.n_positions, self.n_embd)
        self.h = [TransformerBlock(args=args) for _ in range(self.n_layer)]
        self.ln_f = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape

        hidden_states = self.wte(inputs)

        mask = None
        if hidden_states.shape[1] > 1:

            position_ids = mx.array(np.arange(L))
            hidden_states += self.wpe(position_ids)

            mask = create_attention_mask(hidden_states, cache)

        if cache is None:
            cache = [None] * len(self.h)

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPT2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.wte.as_linear(out)
        return out

    def sanitize(self, weights):
        new_weights = {}
        for i in range(self.args.n_layer):
            if f"h.{i}.attn.bias" in weights:
                del weights[f"h.{i}.attn.bias"]
            if f"h.{i}.attn.c_attn.weight" in weights:
                weights[f"h.{i}.attn.c_attn.weight"] = weights[
                    f"h.{i}.attn.c_attn.weight"
                ].transpose(1, 0)
            if f"h.{i}.attn.c_proj.weight" in weights:
                weights[f"h.{i}.attn.c_proj.weight"] = weights[
                    f"h.{i}.attn.c_proj.weight"
                ].transpose(1, 0)
            if f"h.{i}.mlp.c_fc.weight" in weights:
                weights[f"h.{i}.mlp.c_fc.weight"] = weights[
                    f"h.{i}.mlp.c_fc.weight"
                ].transpose(1, 0)
            if f"h.{i}.mlp.c_proj.weight" in weights:
                weights[f"h.{i}.mlp.c_proj.weight"] = weights[
                    f"h.{i}.mlp.c_proj.weight"
                ].transpose(1, 0)
        for weight in weights:
            if not weight.startswith("model."):
                new_weights[f"model.{weight}"] = weights[weight]
            else:
                new_weights[weight] = weights[weight]
        return new_weights

    @property
    def layers(self):
        return self.model.h