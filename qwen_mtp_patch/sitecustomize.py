"""Let the MTP draft of a pruned Qwen3.5/3.6 MoE keep its own expert count (plans/PLAN_QWEN_PRUNE.md §7.1).

vLLM 0.33 builds the MTP draft with the ``vllm_config`` of the target: ``Qwen3_5MTP`` reads ``num_experts`` from the
target config, even when the ``model`` field of the speculative config names another folder for the draft weights.
A target with K experts then loads a draft of 256 experts into tensors of K rows, and the boot fails
("IndexError: index 208 is out of bounds for dimension 0 with size 208").

This patch wraps ``Qwen3_5MTP.__init__``. When the draft config (``speculative_config.draft_model_config``) has another
``num_experts`` than the target, the draft gets a shallow copy of ``vllm_config`` whose text config is a copy with the
expert count of the draft. Nothing else changes, and the patch does nothing when the counts are equal.

Use: put this folder first on ``PYTHONPATH`` of the ``vllm serve`` process. Python imports ``sitecustomize`` at the
start, and the patch applies when vLLM imports ``vllm.model_executor.models.qwen3_5_mtp``.
"""

import copy
import importlib.abc
import importlib.util
import sys

TARGET = "vllm.model_executor.models.qwen3_5_mtp"


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


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)   # one use: find the real module, then wrap its loader
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


if TARGET in sys.modules:
    patch(sys.modules[TARGET])
else:
    sys.meta_path.insert(0, _Finder())
