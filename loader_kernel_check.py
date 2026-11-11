"""The kernel check of plans/PLAN_LOADER_PATCH.md §5.3, on the GPU with real experts.

It loads the experts of one layer from two pruned folders: the K 160 folder (uniform, as the live model) and the MD
folder (a smaller K in that layer, a subset of the same experts). Then it runs the SparkInfer MoE of the patched
exl3.py on both, with the same tokens and the same experts:

1. the MD layer with its own plan and a view of the shared scratch pool, against the K 160 layer with its own scratch;
2. the counts in turn (MD, K 160, MD) on the same pool: the MD output must not change;
3. the plan time and the first run time of each count, and the pool sizes at the live capacity.

Run it with loader_kernel_check.sh, in the server image with the patched exl3.py mounted. It needs about 6 GiB of
GPU memory, and it stops before the load if less than 12 GiB is available (``MemAvailable``: on the GB10 the GPU
allocates from the unified memory, and ``cudaMemGetInfo`` leaves out the page cache that the kernel frees on demand).
"""

import argparse
import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.model_executor.layers.quantization import exl3

H, I, BITS, TOPK = 4096, 2048, 3, 5
DEV = torch.device("cuda", 0)   # the device of the tensors; SparkInfer compares it with the device of the plan
DTYPES = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32}


def available_gib():
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:")) / 2**20


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def read(path, start, info):
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(start + a)
        raw = bytearray(f.read(b - a))
    return torch.frombuffer(raw, dtype=DTYPES[info["dtype"]]).reshape(info["shape"])


def load_layer(folder, layer, experts):
    """The slabs of ``_prepare_rank_sliced_weights`` for ``experts`` (ids in the folder), on the GPU."""
    path = folder / f"exl3-layer-{layer:03d}-tp1-rank0.safetensors"
    start, table = header(path)
    e_n = len(experts)
    s = {"w13": torch.empty((2, e_n, H // 16, I // 16, 16 * BITS), dtype=torch.int16),
         "w2": torch.empty((e_n, I // 16, H // 16, 16 * BITS), dtype=torch.int16),
         **{k: torch.empty((e_n, H), dtype=torch.float16) for k in ("gate_suh", "up_suh", "down_svh")},
         **{k: torch.empty((e_n, I), dtype=torch.float16) for k in ("gate_svh", "up_svh", "down_suh")}}
    for i, e in enumerate(experts):
        t = lambda p, part: read(path, start, table[f"layers.{layer}.ffn.experts.{e}.{p}.rank0.{part}"])  # noqa: E731
        s["w13"][0, i], s["w13"][1, i], s["w2"][i] = t("w1", "trellis"), t("w3", "trellis"), t("w2", "trellis")
        s["gate_suh"][i], s["gate_svh"][i] = t("w1", "suh"), t("w1", "svh")
        s["up_suh"][i], s["up_svh"][i] = t("w3", "suh"), t("w3", "svh")
        s["down_suh"][i], s["down_svh"][i] = t("w2", "suh"), t("w2", "svh")
    marker = read(path, start, table[f"layers.{layer}.ffn.experts.{experts[0]}.w1.rank0.mcg"])
    return {k: v.cuda() for k, v in s.items()}, marker.cuda()


def prepare(api, slabs, marker, tile):
    e_n = slabs["w2"].shape[0]
    plan = api.plan_weights(quant_modes="w4a16", source_format="exl3_trellis_mcg", activation="silu",
                            params_dtype=torch.bfloat16, num_experts=e_n, hidden_size=H, intermediate_size=I,
                            w13_layout="w13", trellis_bits=BITS, trellis_tile_config=tile)
    rot = torch.cat([slabs["gate_svh"], slabs["up_svh"], slabs["down_suh"]], dim=1).contiguous()
    return api.prepare_weights(plan=plan, params_dtype=torch.bfloat16, w1_fp4=slabs["w13"], w2_fp4=slabs["w2"],
                               gate_suh=slabs["gate_suh"], up_suh=slabs["up_suh"], intermediate_rotations=rot,
                               down_svh=slabs["down_svh"], trellis_mcg=marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--uniform", type=Path, required=True, help="the K 160 pruned folder")
    parser.add_argument("--perlayer", type=Path, required=True, help="the MD pruned folder")
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--capacity", type=int, default=8224, help="max_num_batched_tokens of the live server")
    args = parser.parse_args()
    free = available_gib()
    if free < 12:
        raise SystemExit(f"only {free:.1f} GiB memory available; the check needs 12")
    uni = {int(k): v for k, v in json.loads((args.uniform / "PRUNE.json").read_text())["keep"].items()}[args.layer]
    per = {int(k): v for k, v in json.loads((args.perlayer / "PRUNE.json").read_text())["keep"].items()}[args.layer]
    to_uni = [uni.index(e) for e in per]   # MD id -> K 160 id: both keep maps are in the ids of the 216-expert model
    api = exl3._load_sparkinfer_fused_moe()
    tile = exl3.Exl3MoEMethod._trellis_tile_config(H, I)
    report = {"layer": args.layer, "counts": [len(per), len(uni)], "free_gib_before": round(free, 1)}

    t0 = time.time()
    slabs, marker = load_layer(args.uniform, args.layer, list(range(len(uni))))
    w_uni = prepare(api, slabs, marker, tile)
    slabs, marker = load_layer(args.perlayer, args.layer, list(range(len(per))))
    w_per = prepare(api, slabs, marker, tile)
    del slabs
    report["load_s"] = round(time.time() - t0, 1)

    def plan_fn(max_tokens, block_m, weight_plan):
        plan = api.plan(api.Caps(max_tokens=max_tokens, num_topk=TOPK, route_num_experts=0, device=DEV,
                                 weight_plan=weight_plan, quant_mode="w4a16", w4a16_block_size_m=block_m))
        return plan, plan.scratch_specs()[0]

    ns = SimpleNamespace(exl3_layer_bitrates=(BITS,), activation=SimpleNamespace(value="silu"),
                         exl3_params_dtype=torch.bfloat16, exl3_hidden_size=H,
                         exl3_intermediate_size_per_partition=I, exl3_trellis_tile_config=tile)
    roles = {"decode": (32, 8), "prefill": (1024, 64)}   # a small prefill capacity keeps the arena small here
    pool = exl3._plan_scratch_pool(api, ns, sorted({len(per), len(uni)}), plan_fn, roles, DEV)
    report["pool_mib"] = {r: round(t.numel() / 2**20, 1) for r, t in pool.items()}
    report["live_capacity_scratch_mib"] = {
        n: {r: round(exl3._spec_nbytes(plan_fn(m if r == "decode" else args.capacity, b, w.plan)[1]) / 2**20, 1)
            for r, (m, b) in roles.items()} for n, w in ((len(per), w_per), (len(uni), w_uni))}

    g = torch.Generator(device="cuda").manual_seed(0)
    cases = {}
    for role, m in (("decode", 4), ("prefill", 512)):
        x = torch.randn(m, H, generator=g, device="cuda").mul_(0.5).bfloat16()
        ids = torch.stack([torch.randperm(len(per), generator=g, device="cuda")[:TOPK] for _ in range(m)])
        w = torch.rand(m, TOPK, generator=g, device="cuda")
        cases[role] = (x, ids, (w / w.sum(-1, keepdim=True)).float())
    lut = torch.tensor(to_uni, device="cuda")

    def run(weights, role, own_scratch):
        x, ids, w = cases[role]
        ids = ids if weights is w_per else lut[ids]
        m, b = roles[role]
        t = time.time()
        plan, spec = plan_fn(m, b, weights.plan)
        scratch = (torch.empty(spec.shape, dtype=spec.dtype, device=spec.device) if own_scratch
                   else exl3._pool_view(pool[role], spec))
        out = api.run(binding=api.bind(plan, scratch=scratch, a=x, experts=weights, topk_weights=w, topk_ids=ids))
        torch.cuda.synchronize()
        return out.float().clone(), round(time.time() - t, 2)

    results = {}
    for role in roles:
        ref, t_ref = run(w_uni, role, True)
        a, t_a = run(w_per, role, False)
        a2, _ = run(w_per, role, False)
        u, _ = run(w_uni, role, False)    # the K 160 count on the same pool, between two MD runs
        a3, _ = run(w_per, role, False)
        cos = torch.nn.functional.cosine_similarity(a, ref, dim=-1).min().item()
        deterministic = bool(torch.equal(a, a2))
        # with a deterministic kernel the MD output must not change at all; else it must stay within the run noise
        same = torch.equal(a, a3) if deterministic else (
            torch.nn.functional.cosine_similarity(a, a3, dim=-1).min().item() > 0.99999)
        results[role] = {
            "md_equals_k160_bitwise": bool(torch.equal(a, ref)), "md_vs_k160_max_abs": (a - ref).abs().max().item(),
            "md_vs_k160_min_cos": round(cos, 6), "md_repeat_bitwise": deterministic,
            "md_unchanged_after_k160_on_pool": bool(same), "k160_pool_equals_own": bool(torch.equal(u, ref)),
            "first_run_s": {"k160": t_ref, "md": t_a}}
    report["results"] = results
    report["free_gib_after"] = round(available_gib(), 1)
    ok = all(r["md_vs_k160_min_cos"] > 0.9999 and r["md_unchanged_after_k160_on_pool"] for r in results.values())
    report["pass"] = ok
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
