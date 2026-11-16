"""Transformer encoder-decoder from the lab notebook (Attention_Is_All_You_Need.ipynb), with two bug fixes:
  1. PositionalEncoding indexed by batch instead of position (pe[:x.size(0)] on batch-first input).
  2. DecoderLayer reused attn_layer_norm for cross-attention instead of enc_attn_layer_norm.
and one change back to the paper (Vaswani et al. 2017, sec. 3.4) that the notebook omitted:
  3. One embedding matrix shared by encoder, decoder and output projection, scaled by sqrt(d_model).
     Without it, run 26980 learned to ignore the article (uniform cross-attention, same perplexity
     with a shuffled source) and produced generic summaries.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout_rate):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_probs = F.softmax(scores, dim=-1)
        return torch.matmul(self.dropout(attn_probs), value), attn_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(np.sqrt(self.d_k), dropout_rate)

    def split_heads(self, x):
        return x.view(x.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x):
        return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.n_heads * self.d_k)

    def forward(self, query, key, value, mask=None):
        Q, K, V = self.split_heads(self.W_q(query)), self.split_heads(self.W_k(key)), self.split_heads(self.W_v(value))
        if mask is not None:
            mask = mask.unsqueeze(1)  # same mask for all heads
        x, attn = self.attention(Q, K, V, mask)
        return self.W_o(self.group_heads(x)), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.w_1, self.w_2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)  [fix: was (max_len, 1, d_model)]

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])  # [fix: was self.pe[:x.size(0)] -> indexed by batch]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        x1, _ = self.attn_layer(x, x, x, mask)
        x = self.attn_layer_norm(x + self.dropout(x1))
        return self.ff_layer_norm(x + self.dropout(self.ff_layer(x)))


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        x = self.pos_embedding(self.tok_embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, memory, src_mask, tgt_mask):
        x1, _ = self.attn_layer(x, x, x, tgt_mask)
        x = self.attn_layer_norm(x + self.dropout(x1))
        x1, attn = self.enc_attn_layer(x, memory, memory, src_mask)
        x = self.enc_attn_layer_norm(x + self.dropout(x1))  # [fix: notebook used attn_layer_norm here]
        return self.ff_layer_norm(x + self.dropout(self.ff_layer(x))), attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.pos_embedding(self.tok_embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x, attn = layer(x, memory, src_mask, tgt_mask)
        return self.layer_norm(x), attn


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, generator, pad_idx):
        super().__init__()
        self.pad_idx, self.encoder, self.decoder, self.generator = pad_idx, encoder, decoder, generator

    def get_pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(-2)  # (B, 1, L)

    @staticmethod
    def get_subsequent_mask(x):
        L = x.size(1)
        return torch.tril(torch.ones(1, L, L, dtype=torch.bool, device=x.device))  # (1, L, L)

    def forward(self, src, tgt):
        src_mask = self.get_pad_mask(src)
        tgt_mask = self.get_pad_mask(tgt) & self.get_subsequent_mask(tgt)
        dec, attn = self.decoder(tgt, self.encoder(src, src_mask), src_mask, tgt_mask)
        return self.generator(dec), attn

    @torch.no_grad()
    def generate(self, src, bos, eos, max_len):
        """Batched greedy decoding; encoder runs once (the notebook re-encoded every step)."""
        src_mask = self.get_pad_mask(src)
        memory = self.encoder(src, src_mask)
        ys = torch.full((src.size(0), 1), bos, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            dec, _ = self.decoder(ys, memory, src_mask, self.get_subsequent_mask(ys))
            nxt = self.generator(dec[:, -1]).argmax(-1).masked_fill(done, self.pad_idx)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt == eos
            if done.all():
                break
        return ys


def build_model(vocab_size, pad_idx, d_model, n_layers, n_heads, d_ff, dropout):
    enc = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout)
    dec = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout)
    gen = Generator(d_model, vocab_size)
    # shared embedding (paper sec. 3.4): std d_model^-0.5 so that after the sqrt(d_model) scale inputs are ~N(0,1)
    # and initial output logits stay small
    nn.init.normal_(enc.tok_embedding.weight, 0, d_model ** -0.5)
    with torch.no_grad():
        enc.tok_embedding.weight[pad_idx].zero_()
    dec.tok_embedding.weight = enc.tok_embedding.weight
    gen.proj.weight = enc.tok_embedding.weight
    return Transformer(enc, dec, gen, pad_idx)


class NoamOptim:
    """Optimizer wrapper: linear warmup then inverse-sqrt decay (Vaswani et al., 2017)."""

    def __init__(self, optimizer, d_model, factor, n_warmup_steps):
        self.optimizer, self.d_model, self.factor, self.n_warmup_steps, self.n_steps = optimizer, d_model, factor, n_warmup_steps, 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.n_steps += 1
        for g in self.optimizer.param_groups:
            g["lr"] = self.get_lr()
        self.optimizer.step()

    def get_lr(self):
        return self.factor * self.d_model ** -0.5 * min(self.n_steps ** -0.5, self.n_steps * self.n_warmup_steps ** -1.5)
