"""Let the MTP draft of a pruned Qwen3.5/3.6 MoE keep its own expert count (plans/PLAN_QWEN_PRUNE.md §7.1).

vLLM 0.33 builds the MTP draft with the ``vllm_config`` of the target: ``Qwen3_5MTP`` reads ``num_experts`` from the
target config, even when the ``model`` field of the speculative config names another folder for the draft weights.
A target with K experts then loads a draft of 256 experts into tensors of K rows, and the boot fails
("IndexError: index 208 is out of bounds for dimension 0 with size 208").

This patch wraps ``Qwen3_5MTP.__init__``. When the draft config (``speculative_config.draft_model_config``) has another
``num_experts`` than the target, the draft gets a shallow copy of ``vllm_config`` whose text config is a copy with the
expert count of the draft. Nothing else changes, and the patch does nothing when the counts are equal.

``QWEN_EMPTY_CACHE=1`` adds a second patch (§7.7): ``GPUModelRunner.load_model`` ends with ``gc.collect()`` and
``torch.cuda.empty_cache()``. With a pinned KV cache (``--kv-cache-memory-bytes``), vLLM skips the profiling block that
frees the allocator cache, so the blocks that a repack of the weights frees (the Marlin repack of an NVFP4 draft)
stay reserved, and later small allocations pin them. The patch logs the allocated and reserved memory before and
after the release, and after the capture of the CUDA graphs.

Use: put this folder first on ``PYTHONPATH`` of the ``vllm serve`` process. Python imports ``sitecustomize`` at the
start, and the patch applies when vLLM imports ``vllm.model_executor.models.qwen3_5_mtp``.
"""

import copy
import gc
import importlib.abc
import os
import importlib.util
import sys

TARGET = "vllm.model_executor.models.qwen3_5_mtp"
RUNNER = "vllm.v1.worker.gpu_model_runner"


def draft_config(vllm_config):
    """The ``vllm_config`` for the draft: a copy with the expert count of the draft, or the same object."""
    spec = getattr(vllm_config, "speculative_config", None)
    draft = getattr(spec, "draft_model_config", None)
    target = vllm_config.model_config
    if draft is None or draft is target:
        return vllm_config
    n = getattr(draft.hf_text_config, "num_experts", None)
    k = getattr(target.hf_text_config, "num_experts", None)
    if n is None or k is None or n == k:
        return vllm_config
    text = copy.deepcopy(target.hf_text_config)
    text.num_experts = n
    model_config = copy.copy(target)
    model_config.hf_text_config = text
    out = copy.copy(vllm_config)
    out.model_config = model_config
    print(f"qwen_mtp_patch: the MTP draft keeps {n} experts (the target has {k})", file=sys.stderr, flush=True)
    return out


def patch(module):
    cls = module.Qwen3_5MTP
    if getattr(cls.__init__, "_qwen_mtp_patch", False):
        return
    init = cls.__init__

    def __init__(self, *, vllm_config, prefix=""):
        init(self, vllm_config=draft_config(vllm_config), prefix=prefix)

    __init__._qwen_mtp_patch = True
    cls.__init__ = __init__


def memory(torch):
    return f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB"


def patch_runner(module):
    """``load_model`` ends with a release of the allocator cache; ``capture_model`` logs the memory after it."""
    import torch

    cls = module.GPUModelRunner
    if getattr(cls.load_model, "_qwen_mtp_patch", False):
        return
    load, capture = cls.load_model, cls.capture_model

    def load_model(self, *a, **k):
        out = load(self, *a, **k)
        before = memory(torch)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"qwen_mtp_patch: after the load {before}; after empty_cache {memory(torch)}", file=sys.stderr, flush=True)
        return out

    def capture_model(self, *a, **k):
        out = capture(self, *a, **k)
        print(f"qwen_mtp_patch: after the graph capture {memory(torch)}", file=sys.stderr, flush=True)
        return out

    load_model._qwen_mtp_patch = capture_model._qwen_mtp_patch = True
    cls.load_model, cls.capture_model = load_model, capture_model


class _Finder(importlib.abc.MetaPathFinder):
    """Wrap the loader of each module of ``patches`` so that its patch runs after the import; one use per module."""

    def __init__(self, patches):
        self.patches = dict(patches)

    def find_spec(self, name, path, target=None):
        if name not in self.patches:
            return None
        fix = self.patches.pop(name)
        sys.meta_path.remove(self)               # find the real module with this finder off the path
        spec = importlib.util.find_spec(name)
        if self.patches:                         # other modules still wait for their patch
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            fix(module)

        spec.loader.exec_module = exec_module
        return spec


PATCHES = {TARGET: patch, **({RUNNER: patch_runner} if os.environ.get("QWEN_EMPTY_CACHE") == "1" else {})}
for _name in [n for n in PATCHES if n in sys.modules]:
    PATCHES.pop(_name)(sys.modules[_name])
if PATCHES:
    sys.meta_path.insert(0, _Finder(PATCHES))
