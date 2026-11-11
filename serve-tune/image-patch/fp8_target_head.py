"""The FP8 head of the target model (serve-tune/PLAN_SERVE_TUNE.md §7.2, idea #4).

After the weights load, ``install`` makes a rowwise-FP8 copy of the head (``lm_head``, 129,280 x 4,096 BF16,
0.99 GiB) and frees the BF16 weight. The copy becomes the ``quant_method`` of the head, so every logits path reads it:

- the target logits: ``LogitsProcessor._apply_head`` calls ``lm_head.quant_method.apply``;
- the DSpark draft logits: the draft has no head of its own (``has_own_lm_head = False``), so ``load_dspark_model``
  gives it the head module of the target. The draft reads the same FP8 tensor, and no second copy exists.

The kernel is ``fp8_draft_head_logits`` of the image: ``torch._scaled_mm`` with a dynamic scale per token and a
scale per weight row. It has no data-dependent control flow, so the draft can run it inside a CUDA graph.

The output changes, because the verify pass now uses FP8 logits. The replay gate of §7.1 decides.
``FP8_TARGET_HEAD=1`` turns the patch on. The FP8 draft head (``DSPARK_FP8_DRAFT_HEAD``) must stay off: it reads
the BF16 weight, which this patch frees.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fp8_draft_head import fp8_draft_head_logits, quantize_draft_head

logger = init_logger(__name__)
ENV = "FP8_TARGET_HEAD"
GIB = 1 << 30


class Fp8HeadMethod:
    """The quant method of the head after ``install``: the logits from the shared FP8 copy."""

    def __init__(self, head):
        self.head = head

    def apply(self, layer, x, bias=None):
        logits = fp8_draft_head_logits(x, self.head)
        return logits if bias is None else logits + bias

    def embedding(self, layer, input_):
        raise RuntimeError("the FP8 head has no embedding lookup")


def enabled(env=os.environ):
    return env.get(ENV, "0") == "1"


def install(lm_head):
    """Quantize the head, give every logits path the FP8 copy, and free the BF16 weight. Return the bytes freed."""
    if os.environ.get("VLLM_DSPARK_FP8_DRAFT_HEAD", "0") not in ("", "0"):
        raise RuntimeError(f"{ENV}=1 needs DSPARK_FP8_DRAFT_HEAD=0: the FP8 draft head reads the freed BF16 weight")
    weight = lm_head.weight
    head = quantize_draft_head(weight)
    freed = weight.numel() * weight.element_size()
    kept = sum(t.numel() * t.element_size() for t in head)
    lm_head.quant_method = Fp8HeadMethod(head)
    lm_head.weight = torch.nn.Parameter(weight.new_empty(0, weight.shape[1]), requires_grad=False)
    del weight
    torch.cuda.empty_cache()
    logger.info("FP8 target head: %d rows; %.2f GiB BF16 freed, %.2f GiB FP8 kept; the draft shares the FP8 copy",
                head.weight_fp8.shape[0], freed / GIB, kept / GIB)
    return freed
