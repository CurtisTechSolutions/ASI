from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def dump_lru_cache(lru: LRUCache) -> str:
    return f"{lru.capacity}:{str(lru.cache)}"

def load_lru_cache(lru_str: str) -> LRUCache:
    capacity, cache = lru_str.split(":")
    capacity = int(capacity)
    lru = LRUCache(capacity=capacity)
    lru.cache = OrderedDict(cache)
    return lru