"""prune_experts.py on a small synthetic checkpoint: 4 layers (3 hash layers), 8 experts, 6 kept, 10 tokens."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import mask_checkpoint as mc
import prune_experts as pe
from test_topk_checkpoint import bf16, save, write_manifest

VOCAB, HIDDEN, EXPERTS, LAYERS = 10, 4, 8, 4
DROP = {0: [1, 4], 1: [0, 7], 2: [2, 3], 3: [5, 6]}
VARIED = {0: [1], 1: [0], 2: [2, 3, 6], 3: [5, 6]}  # K 7, 7, 5, 6: a different K per layer
ORIGINAL = [0, 1, 3, 4, 5, 7, 8, 9]  # the 8 experts kept from 10 by the first prune


def expert_tensors(layer):
    out = {}
    for e in range(EXPERTS):
        for proj in ("w1", "w2"):
            out[f"layers.{layer}.ffn.experts.{e}.{proj}.rank0.trellis"] = (
                "I32", np.full((2, 3), layer * 100 + e * 10 + int(proj[1]), np.int32))
            out[f"layers.{layer}.ffn.experts.{e}.{proj}.rank0.mcg"] = ("I32", np.array(e, np.int32))
    return out


def checkpoint(base, drop=DROP):
    rng = np.random.default_rng(0)
    carried = {"embed.weight": ("BF16", bf16(rng.standard_normal((VOCAB, HIDDEN))))}
    for i in range(LAYERS):
        carried[f"layers.{i}.ffn.gate.weight"] = ("BF16", bf16(rng.standard_normal((EXPERTS, HIDDEN))))
        if i < mc.HASH_LAYERS:
            kept = [e for e in range(EXPERTS) if e not in drop[i]]
            carried[f"layers.{i}.ffn.gate.tid2eid"] = (
                "I64", np.array([rng.permutation(kept)[:5] for _ in range(VOCAB)], np.int64))
    bias = rng.standard_normal(EXPERTS).astype(np.float32)
    bias[drop[3]] = mc.MASK_BIAS
    carried["layers.3.ffn.gate.bias"] = ("F32", bias)
    carried["mtp.0.ffn.gate.weight"] = ("BF16", bf16(rng.standard_normal((EXPERTS, HIDDEN))))
    carried["hc_head_base"] = ("F32", np.arange(3, dtype=np.float32))  # 12 bytes: later offsets not 8-aligned
    save(base / "carried-001.safetensors", carried)
    save(base / "carried-002.safetensors", {"other": ("F32", np.ones(3, np.float32))})
    weight_map = {**{k: "carried-001.safetensors" for k in carried}, "other": "carried-002.safetensors"}
    for i in range(LAYERS):
        name = f"exl3-layer-{i:03d}-tp1-rank0.safetensors"
        tensors = expert_tensors(i)
        save(base / name, tensors)
        weight_map.update({k: name for k in tensors})
    (base / pe.INDEX).write_text(json.dumps({"metadata": {"total_size": 1000}, "weight_map": weight_map}))
    (base / "config.json").write_text(json.dumps({
        "n_routed_experts": EXPERTS, "num_experts_per_tok": 5,
        "hybrid_tr3_tail": {"experts_per_layer": EXPERTS}, "reap_pruning": {"kept_experts_per_layer": EXPERTS}}))
    (base / "quantization_config.json").write_text(json.dumps({"rank_sliced": {"experts_per_layer": EXPERTS}}))
    tiers = {str(i): {"k": [3] * (EXPERTS - 1) + [4]} for i in range(LAYERS)}
    tiers.update(experts_per_layer=EXPERTS, counts={"3": 28, "4": 4}, total_expert_triplets=32, reap_plan_sha256="x")
    (base / pe.TIERS).write_text(json.dumps(tiers))
    (base / pe.REAP).write_text(json.dumps({"kept_experts_per_layer": EXPERTS, "keep_maps": {
        "keep_count": EXPERTS, "mtp_keep": ORIGINAL, "mtp_ranked": list(range(10)),
        "keep_by_layer": {str(i): ORIGINAL for i in range(LAYERS)},
        "ranked_by_layer": {str(i): list(reversed(range(10))) for i in range(LAYERS)}}}))
    (base / "README.md").write_text("model card")
    write_manifest(base)
    manifest = json.loads((base / pe.MANIFEST).read_text())
    manifest["tensor_count"] = 999
    (base / pe.MANIFEST).write_text(json.dumps(manifest))


def mask_folder(base, mask, drop=DROP):
    """Like the pick: a real carried-001 (the mask edits), a real config, symlinks to the rest."""
    mask.mkdir()
    for p in sorted(base.iterdir()):
        if p.name in ("carried-001.safetensors", "config.json", pe.MANIFEST):
            (mask / p.name).write_bytes(p.read_bytes())
        else:
            (mask / p.name).symlink_to(p)
    (mask / "MASK.json").write_text(json.dumps({"keep": 6, "layers": {
        str(i): {"drop": d, "drop_original": [ORIGINAL[e] for e in d]} for i, d in drop.items()}}))
    (mask / "TOPK.json").write_text('{"top_k": 5}')


class PruneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base, self.mask, self.out = self.root / "tp1", self.root / "mask", self.root / "prune"
        self.base.mkdir()
        checkpoint(self.base)
        mask_folder(self.base, self.mask)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return pe.build(self.mask, self.out, base=self.base)

    def kept(self, layer):
        return [e for e in range(EXPERTS) if e not in DROP[layer]]

    def test_the_kept_experts_keep_their_bytes_under_the_new_ids(self):
        record = self.build()
        self.assertEqual(record["kept_experts"], 6)
        self.assertEqual(record["tensors_removed"], LAYERS * 2 * 4)
        for i in range(LAYERS):
            path = self.out / f"exl3-layer-{i:03d}-tp1-rank0.safetensors"
            self.assertFalse(path.is_symlink())
            _, table = mc.header(path)
            self.assertEqual(len(table) - 1, 6 * 4)
            for new, old in enumerate(self.kept(i)):
                name = f"layers.{i}.ffn.experts.{new}.w2.rank0.trellis"
                self.assertTrue((mc.read_tensor(path, name) == i * 100 + old * 10 + 2).all(), name)
                self.assertEqual(int(mc.read_tensor(path, f"layers.{i}.ffn.experts.{new}.w1.rank0.mcg")), old)
            self.assertNotIn(f"layers.{i}.ffn.experts.6.w1.rank0.mcg", table)

    def test_the_router_follows_the_keep_order(self):
        self.build()
        src, dst = self.mask / "carried-001.safetensors", self.out / "carried-001.safetensors"
        for i in range(LAYERS):
            name = f"layers.{i}.ffn.gate.weight"
            self.assertTrue(np.array_equal(mc.read_tensor(dst, name), mc.read_tensor(src, name)[self.kept(i)]))
        for i in range(mc.HASH_LAYERS):
            name = f"layers.{i}.ffn.gate.tid2eid"
            new = mc.read_tensor(dst, name)
            self.assertEqual(new.shape, (VOCAB, 5))
            self.assertTrue((new < 6).all())
            self.assertTrue(np.array_equal(np.array(self.kept(i))[new], mc.read_tensor(src, name)))
        bias = mc.read_tensor(dst, "layers.3.ffn.gate.bias")
        self.assertEqual(bias.tolist(), mc.read_tensor(src, "layers.3.ffn.gate.bias")[self.kept(3)].tolist())
        self.assertFalse((bias == mc.MASK_BIAS).any())
        for name in ("mtp.0.ffn.gate.weight", "embed.weight"):
            self.assertTrue(np.array_equal(mc.read_tensor(dst, name), mc.read_tensor(src, name)), name)

    def test_the_json_files_and_the_links(self):
        record = self.build()
        cfg = json.loads((self.out / "config.json").read_text())
        self.assertEqual((cfg["n_routed_experts"], cfg["hybrid_tr3_tail"]["experts_per_layer"],
                          cfg["reap_pruning"]["kept_experts_per_layer"], cfg["num_experts_per_tok"]), (6, 6, 6, 5))
        self.assertNotIn("n_routed_experts_per_layer", cfg)  # a uniform folder loads in the stock image
        quant = json.loads((self.out / "quantization_config.json").read_text())
        self.assertEqual(quant["rank_sliced"]["experts_per_layer"], cfg["n_routed_experts"])  # the vLLM rule
        index = json.loads((self.out / pe.INDEX).read_text())
        self.assertIn("layers.2.ffn.experts.5.w1.rank0.trellis", index["weight_map"])
        self.assertNotIn("layers.2.ffn.experts.6.w1.rank0.trellis", index["weight_map"])
        self.assertEqual(index["metadata"]["total_size"], 1000 + record["tensor_bytes_change"])
        self.assertLess(record["tensor_bytes_change"], 0)
        tiers = json.loads((self.out / pe.TIERS).read_text())
        self.assertEqual(tiers["3"]["k"], [3, 3, 3, 3, 3, 4])  # drops 5 and 6, keeps the 4-bit expert 7
        self.assertEqual(tiers["0"]["k"], [3, 3, 3, 3, 3, 4])
        self.assertEqual((tiers["experts_per_layer"], tiers["total_expert_triplets"], tiers["counts"]),
                         (6, 24, {"3": 21, "4": 3}))  # layer 1 drops the 4-bit expert 7
        plan = json.loads((self.out / "REAP_K6_PLAN.json").read_text())
        self.assertEqual(plan["keep_maps"]["keep_by_layer"]["3"], [ORIGINAL[e] for e in self.kept(3)])
        self.assertEqual(plan["keep_maps"]["mtp_keep"], ORIGINAL)
        self.assertEqual(plan["keep_maps"]["ranked_by_layer"]["0"], [9, 8, 7, 4, 3, 0])
        self.assertEqual(tiers["reap_plan_sha256"], mc.sha256_file(self.out / "REAP_K6_PLAN.json"))
        manifest = json.loads((self.out / pe.MANIFEST).read_text())
        for item in manifest["files"]:
            self.assertEqual(item["bytes"], (self.out / item["name"]).stat().st_size)
            self.assertEqual(item["sha256"], hashlib.sha256((self.out / item["name"]).read_bytes()).hexdigest())
        self.assertEqual(manifest["tensor_count"], 999 - record["tensors_removed"])
        self.assertTrue((self.out / "carried-002.safetensors").is_symlink())
        self.assertEqual(Path(os.readlink(self.out / "README.md")), (self.base / "README.md").resolve())
        self.assertTrue((self.out / pe.REAP).is_symlink())
        self.assertEqual((self.out / "TOPK.json").read_text(), '{"top_k": 5}')
        prune = json.loads((self.out / "PRUNE.json").read_text())
        self.assertEqual(prune["self_check"]["verify_tp1_manifest"]["tensor_files"], 6)
        self.assertEqual(prune["self_check"]["experts_same_bytes"], LAYERS * 6 * 4)
        self.assertEqual(prune["self_check"]["router_tensors"], LAYERS * 2)
        with self.assertRaises(FileExistsError):
            self.build()

    def test_a_different_k_per_layer_writes_the_lists_of_the_loader_patch(self):
        base, mask, out = self.root / "tp1-varied", self.root / "mask-varied", self.root / "prune-varied"
        base.mkdir()
        checkpoint(base, VARIED)
        mask_folder(base, mask, VARIED)
        record = pe.build(mask, out, base=base)
        ks = [7, 7, 5, 6]
        self.assertEqual(record["kept_experts"], ks)
        self.assertEqual(record["tensors_removed"], 7 * 2 * 2)
        cfg = json.loads((out / "config.json").read_text())
        self.assertEqual((cfg["n_routed_experts"], cfg["n_routed_experts_per_layer"]), (7, ks))
        self.assertEqual((cfg["hybrid_tr3_tail"]["experts_per_layer"], cfg["reap_pruning"]["kept_experts_per_layer"]),
                         (ks, ks))
        self.assertEqual(json.loads((out / "quantization_config.json").read_text())["rank_sliced"]["experts_per_layer"],
                         ks)
        tiers = json.loads((out / pe.TIERS).read_text())
        self.assertEqual((tiers["experts_per_layer"], tiers["total_expert_triplets"], len(tiers["2"]["k"])),
                         (ks, 25, 5))
        plan = json.loads((out / "REAP_K6_PLAN.json").read_text())  # the rounded mean of 6.25
        self.assertEqual((plan["kept_experts_per_layer"], plan["keep_maps"]["keep_count"]), (ks, ks))
        self.assertEqual(plan["keep_maps"]["keep_by_layer"]["2"], [ORIGINAL[e] for e in (0, 1, 4, 5, 7)])
        _, table = mc.header(out / "exl3-layer-002-tp1-rank0.safetensors")
        self.assertEqual(len(table) - 1, 5 * 4)
        self.assertEqual(int(mc.read_tensor(out / "exl3-layer-002-tp1-rank0.safetensors",
                                            "layers.2.ffn.experts.4.w1.rank0.mcg")), 7)
        gate = mc.read_tensor(out / "carried-001.safetensors", "layers.1.ffn.gate.weight")
        self.assertEqual(gate.shape, (7, HIDDEN))
        self.assertEqual(record["self_check"]["experts_same_bytes"], sum(ks) * 4)
        cfg["n_routed_experts_per_layer"] = [7, 7, 6, 5]
        (out / "config.json").write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "expert counts do not match"):
            pe.self_check(mask, out)

    def test_a_dropped_id_in_tid2eid_stops_the_tool(self):
        path = self.mask / "carried-001.safetensors"
        tid = mc.read_tensor(path, "layers.1.ffn.gate.tid2eid")
        tid[3, 2] = 7
        mc.write_tensor(path, "layers.1.ffn.gate.tid2eid", tid)
        with self.assertRaisesRegex(ValueError, "holds a dropped id"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_a_bias_that_is_not_masked_stops_the_tool(self):
        path = self.mask / "carried-001.safetensors"
        bias = mc.read_tensor(path, "layers.3.ffn.gate.bias")
        bias[5] = 0.5
        mc.write_tensor(path, "layers.3.ffn.gate.bias", bias)
        with self.assertRaisesRegex(ValueError, "masked bias"):
            self.build()

    def test_the_self_check_finds_a_changed_byte(self):
        self.build()
        pe.self_check(self.mask, self.out)
        path = self.out / "exl3-layer-002-tp1-rank0.safetensors"
        start, table = mc.header(path)
        raw = bytearray(path.read_bytes())
        raw[start + table["layers.2.ffn.experts.4.w1.rank0.trellis"]["data_offsets"][0]] ^= 1
        path.write_bytes(bytes(raw))
        with self.assertRaisesRegex(ValueError, "experts.4.w1.rank0.trellis: the bytes changed"):
            pe.self_check(self.mask, self.out)


if __name__ == "__main__":
    unittest.main()
