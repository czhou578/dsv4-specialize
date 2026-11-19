"""A memo for methods. Each method keeps its results in its own slot of the object."""

def memo(fn):
    slot = "_memo_" + fn.__name__

    def wrapper(self, *args):
        cache = self.__dict__.setdefault(slot, {})
        if args not in cache:
            cache[args] = fn(self, *args)
        return cache[args]

    return wrapper

def invalidate(obj):
    """Forget all memo results of obj."""
    obj.__dict__.pop("_memo", None)
