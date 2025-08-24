from __future__ import annotations
from functools import lru_cache

# simple in-process cache via lru_cache; ok for single-process Dash
def memoize(maxsize: int = 32):
    def deco(fn):
        return lru_cache(maxsize=maxsize)(fn)
    return deco
