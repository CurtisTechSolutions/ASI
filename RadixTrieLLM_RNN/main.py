import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


VOCAB_SIZE = 256  # byte-level vocab
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2


class NodeTransformer(nn.Module):
    """Small transformer encoder attached to a single trie node.

    Encodes the edge-label sequence (bytes) plus optional incoming context
    and produces a pooled representation + next-token logits.
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor,
                context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_ids: (batch, seq_len) integer byte ids for this node's edge label
        context: optional (batch, d_model) pooled state from parent node
        returns: (pooled_state, next_token_logits)
        """
        b, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.embed(token_ids) + self.pos_embed(positions)

        if context is not None:
            x = x + context.unsqueeze(1)  # inject parent context into every position

        x = self.norm(x)
        encoded = self.encoder(x)
        pooled = encoded[:, -1, :]  # use last position as summary
        logits = self.out_proj(pooled)
        return pooled, logits


@dataclass
class RadixTrieNode:
    label: str = ""  # edge label leading into this node
    children: Dict[str, "RadixTrieNode"] = field(default_factory=dict)
    is_end: bool = False
    transformer: NodeTransformer = field(default_factory=NodeTransformer)

    def encode_label(self) -> torch.Tensor:
        if not self.label:
            return torch.zeros((1, 1), dtype=torch.long)
        ids = [b for b in self.label.encode("utf-8")]
        return torch.tensor([ids], dtype=torch.long)


class RadixTrie:
    def __init__(self):
        self.root = RadixTrieNode(label="")

    # ---------- Insertion (standard radix trie splitting logic) ----------

    def insert(self, word: str):
        node = self.root
        remaining = word

        while True:
            match_child_key = None
            common_len = 0

            for key, child in node.children.items():
                common_len = self._common_prefix_len(remaining, key)
                if common_len > 0:
                    match_child_key = key
                    break

            if match_child_key is None:
                # no overlap: create new leaf child
                new_node = RadixTrieNode(label=remaining, is_end=True)
                node.children[remaining] = new_node
                return

            child = node.children[match_child_key]

            if common_len == len(match_child_key):
                # full match of child's label; descend
                remaining = remaining[common_len:]
                if remaining == "":
                    child.is_end = True
                    return
                node = child
                continue

            # partial match: split existing child
            common_prefix = match_child_key[:common_len]
            child_suffix = match_child_key[common_len:]

            split_node = RadixTrieNode(label=common_prefix)
            child.label = child_suffix
            split_node.children[child_suffix] = child

            del node.children[match_child_key]
            node.children[common_prefix] = split_node

            remaining = remaining[common_len:]
            if remaining == "":
                split_node.is_end = True
                return

            node = split_node

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    # ---------- Search ----------

    def search(self, word: str) -> bool:
        node = self.root
        remaining = word

        while remaining:
            for key, child in node.children.items():
                if remaining.startswith(key):
                    remaining = remaining[len(key):]
                    node = child
                    break
            else:
                return False

        return node.is_end

    # ---------- Transformer-driven forward pass over the matched path ----------

    def encode_path(self, word: str) -> Optional[torch.Tensor]:
        """Run each node's transformer along the path matching `word`,
        chaining pooled context from parent to child. Returns final
        pooled representation, or None if the path doesn't fully match.
        """
        node = self.root
        remaining = word
        context: Optional[torch.Tensor] = None

        while remaining:
            matched_child = None
            for key, child in node.children.items():
                if remaining.startswith(key):
                    matched_child = child
                    remaining = remaining[len(key):]
                    break

            if matched_child is None:
                return None

            token_ids = matched_child.encode_label()
            context, _ = matched_child.transformer(token_ids, context)
            node = matched_child

        return context

    def predict_next_byte(self, word: str) -> Optional[int]:
        """Uses the last node's transformer logits to predict the next byte."""
        node = self.root
        remaining = word
        context: Optional[torch.Tensor] = None
        last_logits: Optional[torch.Tensor] = None

        while remaining:
            matched_child = None
            for key, child in node.children.items():
                if remaining.startswith(key):
                    matched_child = child
                    remaining = remaining[len(key):]
                    break

            if matched_child is None:
                return None

            token_ids = matched_child.encode_label()
            context, last_logits = matched_child.transformer(token_ids, context)
            node = matched_child

        if last_logits is None:
            return None

        return int(torch.argmax(last_logits, dim=-1).item())


if __name__ == "__main__":
    trie = RadixTrie()
    words = ["hello", "help", "helium", "hero", "her"]
    for w in words:
        trie.insert(w)

    for w in ["hello", "help", "he", "hero"]:
        print(w, "found:", trie.search(w))

    with torch.no_grad():
        pooled = trie.encode_path("hello")
        print("pooled shape:", None if pooled is None else pooled.shape)

        predicted_byte = trie.predict_next_byte("hel")
        print("predicted next byte after 'hel':", predicted_byte)
