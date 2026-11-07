"""mask_checkpoint.py on a small synthetic checkpoint: 4 layers (3 hash layers), 8 of 10 experts kept."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import mask_checkpoint as mc

KEPT = [0, 1, 2, 3, 5, 6, 8, 9]  # original ids; the checkpoint numbers them 0-7
RANKED = [9, 1, 5, 0, 7, 3, 8, 2, 6, 4]  # best first; 7 and 4 were pruned before


def save(path, tensors):
    table, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        blob = array.tobytes()
        table[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    head = json.dumps(table).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def checkpoint(base):
    rng = np.random.default_rng(0)
    gate = rng.standard_normal((8, 4)).astype(np.float32)
    gate[6] = gate[1] * 2  # current expert 6 (original 8) points like expert 1
    bf16 = (gate.view(np.uint32) >> 16).astype(np.uint16)
    carried = {f"layers.{i}.ffn.gate.weight": ("BF16", bf16) for i in range(4)}
    for i in range(3):
        carried[f"layers.{i}.ffn.gate.tid2eid"] = ("I64", np.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 6, 7], [2, 3, 4]], np.int64))
    carried["layers.3.ffn.gate.bias"] = ("F32", np.full(8, 8.5, np.float32))
    save(base / "carried-001.safetensors", carried)
    save(base / "carried-002.safetensors", {"other": ("F32", np.ones(3, np.float32))})
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        **{k: "carried-001.safetensors" for k in carried}, "other": "carried-002.safetensors"}}))
    layers = {str(i): KEPT for i in range(4)}
    (base / "REAP_K216_PLAN.json").write_text(json.dumps({"keep_maps": {
        "keep_by_layer": layers, "ranked_by_layer": {str(i): RANKED for i in range(4)},
        "structured_ranked_by_category": {"tools": {str(i): [3, 6] for i in range(4)},
                                          "agentic": {str(i): [6, 9] for i in range(4)}}}}))
    files = [{"name": p.name, "bytes": p.stat().st_size,
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(base.glob("*.safetensors"))]
    (base / "rank-sliced-tp1-manifest.json").write_text(json.dumps({"format": "x", "files": files}))
    (base / "config.json").write_text("{}")


@mock.patch.object(mc, "MANDATORY_PER_CATEGORY", 1)  # mandatory: original 3 and 6
class MaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name) / "tp1"
        self.base.mkdir()
        checkpoint(self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def test_drop_the_lowest_rank_and_keep_the_mandatory_experts(self):
        plan = json.loads((self.base / "REAP_K216_PLAN.json").read_text())
        drops = mc.drop_lists(plan, 5)
        self.assertEqual(drops[0], ([2, 6, 0], [2, 8, 0]))  # (current ids, original ids)
        with self.assertRaises(ValueError):
            mc.drop_lists(plan, 1)  # 2 mandatory experts per layer; K=1 leaves no room

    def test_the_bot_rank_drops_the_least_used_and_keeps_the_mandatory_experts(self):
        plan = json.loads((self.base / "REAP_K216_PLAN.json").read_text())
        rank = {"layers": {str(i): [{"id": e} for e in [7, 6, 5, 4, 3, 2, 1, 0]] for i in range(4)}}
        bot = mc.bot_order(rank)
        self.assertEqual(bot[2], [7, 6, 5, 4, 3, 2, 1, 0])
        # least used first: 0, 1, 2, then 3 is mandatory (original 3), so 4
        self.assertEqual(mc.drop_lists(plan, 4, bot)[0], ([0, 1, 2, 4], [0, 1, 2, 5]))
        bot[1] = [7, 6, 5, 4, 3, 2, 1, 1]
        with self.assertRaises(ValueError):
            mc.drop_lists(plan, 4, bot)  # not a permutation of the current ids

    def test_build_with_the_bot_rank_records_the_rank_file(self):
        rank = Path(self.tmp.name) / "expert_rank_bot.json"
        rank.write_text(json.dumps({"layers": {str(i): [{"id": e} for e in range(8)] for i in range(4)}}))
        record = mc.build(self.base, 6, Path(self.tmp.name) / "bot", dry_run=True, rank=rank)
        self.assertEqual(record["layers"][3]["drop"], [7, 6])
        self.assertEqual(record["rank"]["sha256"], hashlib.sha256(rank.read_bytes()).hexdigest())
        self.assertEqual(mc.build(self.base, 6, Path(self.tmp.name) / "reap", dry_run=True)["rank"],
                         "REAP_K216_PLAN.json ranked_by_layer")

    def test_hash_entries_move_to_the_most_similar_kept_expert(self):
        tid = np.array([[0, 1, 6], [6, 2, 3]], np.int64)
        gate = np.eye(8, dtype=np.float32)
        gate[6] = [0, 0.9, 0.1, 0, 0, 0, 0, 0]  # 6 is nearest to 1, then 2
        new, n = mc.remap_hash(tid, [6], mc.similarity_order(gate, [6]))
        self.assertEqual(new.tolist(), [[0, 1, 2], [1, 2, 3]])  # row 0 holds 1, so 2
        self.assertEqual(n, 2)
        crowded = np.array([[6, 0]] * 4, np.int64)  # cap: 2 * 8 entries / 7 kept = 2.3
        new, _ = mc.remap_hash(crowded, [6], mc.similarity_order(gate, [6]))
        self.assertEqual(new[:, 0].tolist(), [1, 1, 1, 2])  # 1 is full after 3

    def test_build_writes_symlinks_edited_copies_and_the_manifest(self):
        out = Path(self.tmp.name) / "mask"
        self.assertEqual(mc.build(self.base, 6, out, dry_run=True)["edited_files"],
                         ["carried-001.safetensors"])
        self.assertFalse(out.exists())
        record = mc.build(self.base, 6, out)
        self.assertEqual(record["layers"][3]["drop_original"], [2, 8])
        self.assertTrue((out / "carried-002.safetensors").is_symlink())
        self.assertTrue((out / "config.json").is_symlink())
        self.assertFalse((out / "carried-001.safetensors").is_symlink())
        bias = mc.read_tensor(out / "carried-001.safetensors", "layers.3.ffn.gate.bias")
        self.assertEqual(np.flatnonzero(bias < 0).tolist(), [2, 6])
        tid = mc.read_tensor(out / "carried-001.safetensors", "layers.0.ffn.gate.tid2eid")
        self.assertFalse(np.isin(tid, [2, 6]).any())
        self.assertTrue(all(len(set(row)) == len(row) for row in tid.tolist()))
        self.assertEqual(tid[2].tolist(), [1, 7, 0])  # 6 points like 1
        same = mc.read_tensor(self.base / "carried-001.safetensors", "layers.0.ffn.gate.weight")
        self.assertTrue(np.array_equal(
            mc.read_tensor(out / "carried-001.safetensors", "layers.0.ffn.gate.weight"), same))
        manifest = json.loads((out / "rank-sliced-tp1-manifest.json").read_text())
        for item in manifest["files"]:
            self.assertEqual(item["sha256"], hashlib.sha256((out / item["name"]).read_bytes()).hexdigest())
        self.assertEqual(json.loads((out / "MASK.json").read_text())["keep"], 6)


if __name__ == "__main__":
    unittest.main()
