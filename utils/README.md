# utils

Small, dependency-free helpers shared by the experiments in this repository.
Nothing here is specific to one architecture; anything that is lives with the
architecture that needs it.

## Contents

| file | what it is |
|---|---|
| `lru_cache.py` | a least-recently-used cache over `OrderedDict` — `get` (returns `-1` on a miss), `put` (evicts the oldest entry past `capacity`), plus `dump_lru_cache` / `load_lru_cache`, intended for writing a cache out as a string and reading it back |

## Using it

```python
from utils.lru_cache import LRUCache, dump_lru_cache, load_lru_cache

cache = LRUCache(capacity=2)
cache.put("a", 1)
cache.put("b", 2)
cache.get("a")          # 1, and "a" becomes the most recent
cache.put("c", 3)       # evicts "b"
cache.get("b")          # -1
```

## A caveat on persistence

`dump_lru_cache` serialises with `str(OrderedDict)`, which produces
`"2:OrderedDict([('a', 1), ('b', 2)])"`. `load_lru_cache` then splits that on
`":"` and hands the remainder straight to `OrderedDict(...)`, so the round trip
does not currently work — the split yields three parts and the call raises
`ValueError`. Use `json`/`pickle` if you need the cache to survive a process.
