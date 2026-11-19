from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("negative capacity")
        self.capacity, self.d = capacity, OrderedDict()
    def get(self, key, default=None):
        if key in self.d:
            self.d.move_to_end(key)
        return self.d.get(key, default)
    def peek(self, key, default=None):
        return self.d.get(key, default)
    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        elif self.capacity == 0:
            return
        elif len(self.d) >= self.capacity:
            self.d.popitem(last=False)
        self.d[key] = value
    def __len__(self):
        return len(self.d)
    def keys(self):
        return list(self.d)
