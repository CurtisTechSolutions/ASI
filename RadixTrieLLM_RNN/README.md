# RadixTrieLLM_RNN

A radix trie where **every node owns its own small neural network**, and a path
through the trie is the sequence. Chosen specifically to focus on information
compression: a shared prefix is stored once, so the structure does the
compressing and each node's model only has to explain the bytes on its own edge.

This is one variation of many. `RadixCyclicNN/` is where the idea is built out
properly — with cycles, a self-compressing graph and a sine activation instead
of a trie of transformers.

## Contents

| file | what it is |
|---|---|
| `main.py` | the whole thing: `NodeTransformer`, `RadixTrieNode`, `RadixTrie`, and a demo under `__main__` |

### What is in `main.py`

| piece | what it does |
|---|---|
| `NodeTransformer` | a small byte-level transformer encoder (`d_model=64`, 4 heads, 2 layers, 256-symbol vocab) attached to a single trie node. Encodes that node's edge label plus optional context handed down from the parent, and returns a pooled representation and next-token logits. |
| `RadixTrieNode` | one node: its incoming edge label, its children, whether a word ends here, and its own `NodeTransformer`. |
| `RadixTrie` | `insert` with standard radix splitting (a partial prefix match splits the existing child in two), `search`, `encode_path` (walk the trie and thread the pooled state from parent to child), and `predict_next_byte`. |

Running `python3 main.py` inserts `hello, help, helium, hero, her`, searches
them, and prints the pooled shape for `"hello"` and the predicted next byte
after `"hel"`.

## Running it

```bash
pip install torch
python3 main.py
```

## Note on the naming

`RadixTreeLLM/` is the companion directory and its README carries this
directory's name — the two were written together and the radix *tree* / radix
*trie* distinction is the one made in `RadixCyclicNN/DESIGN.md` §5.2: path
compression from the radix tree, terminal-on-internal nodes from the trie.
