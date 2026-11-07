"""topk_checkpoint.py on a small synthetic checkpoint: 10 tokens, 8 hidden units, 12 experts, 3 hash layers."""

import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import mask_checkpoint as mc
import topk_checkpoint as tk

VOCAB, HIDDEN, EXPERTS = 10, 8, 12
CONFIG = '{\n  "hidden_size": 8,\n  "num_experts_per_tok": 6,\n  "n_routed_experts": 12\n}\n'


def bf16(array):
    return (np.asarray(array, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def save(path, tensors):
    table, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name, (dtype, array) in tensors.items():
        blob = array.tobytes()
        table[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    head = json.dumps(table).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def checkpoint(base):
    rng = np.random.default_rng(0)
    carried = {}
    for i in range(3):
        rows = np.array([rng.permutation(EXPERTS)[:6] for _ in range(VOCAB)], np.int64)
        carried[f"layers.{i}.ffn.gate.tid2eid"] = ("I64", rows)
    carried["hc_head_base"] = ("F32", np.arange(3, dtype=np.float32))  # 12 bytes: later offsets not 8-aligned
    carried["embed.weight"] = ("BF16", bf16(rng.standard_normal((VOCAB, HIDDEN))))
    for i in range(4):
        carried[f"layers.{i}.ffn_norm.weight"] = ("BF16", bf16(rng.uniform(0.5, 1.5, HIDDEN)))
        carried[f"layers.{i}.ffn.gate.weight"] = ("BF16", bf16(rng.standard_normal((EXPERTS, HIDDEN))))
    carried["layers.3.ffn.gate.bias"] = ("F32", np.zeros(EXPERTS, np.float32))
    carried["big"] = ("F32", rng.standard_normal(1000).astype(np.float32))
    save(base / "carried-001.safetensors", carried)
    save(base / "carried-002.safetensors", {"other": ("F32", np.ones(3, np.float32))})
    save(base / "exl3-layer-000-tp1-rank0.safetensors", {"q": ("F32", np.ones(5, np.float32))})
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        **{k: "carried-001.safetensors" for k in carried}, "other": "carried-002.safetensors",
        "q": "exl3-layer-000-tp1-rank0.safetensors"}}))
    (base / "config.json").write_text(CONFIG)
    write_manifest(base)


def write_manifest(folder):
    files = [{"bytes": p.stat().st_size, "name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(folder.glob("*.safetensors"))]
    data = sum(p.stat().st_size - mc.header(p)[0] for p in folder.glob("*.safetensors"))
    (folder / tk.MANIFEST).write_text(json.dumps(
        {"files": files, "format": "rank-sliced-exl3-tp1-v1", "target_tp": 1, "tensor_bytes": data}))


def mask_folder(base, mask):
    """Like mask_checkpoint.py: edited real copies of carried-001/002, symlinks to the rest."""
    mask.mkdir()
    for p in sorted(base.iterdir()):
        if p.name in ("carried-001.safetensors", "carried-002.safetensors", tk.MANIFEST):
            (mask / p.name).write_bytes(p.read_bytes())
        else:
            (mask / p.name).symlink_to(p)
    (mask / "MASK.json").write_text('{"keep": 10}')
    write_manifest(mask)


class TopkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "tp1"
        self.base.mkdir()
        checkpoint(self.base)
        self.mask = self.root / "mask"
        mask_folder(self.base, self.mask)
        self.rank = self.root / "expert_rank_bot.json"
        self.rank.write_text(json.dumps({"layers": {str(i): [{"id": e} for e in range(EXPERTS)] for i in range(4)}}))

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, src, top_k, name):
        return tk.build(src, top_k, self.root / name, rank=self.rank, base=self.base)

    def test_the_drop_rule_drops_the_lowest_scores_and_keeps_the_column_order(self):
        tid = np.array([[5, 3, 9, 1, 7, 2], [0, 1, 2, 3, 4, 5]], np.int64)
        scores = np.array([[0.5, -1.0, 2.0, 0.1, 3.0, 0.2], [1, 1, 1, 1, 1, 1]], np.float32)
        self.assertEqual(tk.cut_rows(tid, scores, 5).tolist(), [[5, 9, 1, 7, 2], [1, 2, 3, 4, 5]])
        self.assertEqual(tk.cut_rows(tid, scores, 4).tolist(), [[5, 9, 7, 2], [2, 3, 4, 5]])  # ties: left first

    def test_the_tables_match_a_full_computation_across_chunks(self):
        read = lambda name: mc.read_tensor(self.base / "carried-001.safetensors", name)
        x = mc.bf16_to_f32(read("embed.weight"))
        x = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + tk.RMS_EPS)
        with mock.patch.object(tk, "CHUNK_ROWS", 3):  # chunks of 3, 3, 3, 1 rows
            tables, share = tk.new_tables(self.base, 5)
        for i in range(3):
            logits = (x * mc.bf16_to_f32(read(f"layers.{i}.ffn_norm.weight"))) @ \
                mc.bf16_to_f32(read(f"layers.{i}.ffn.gate.weight")).T
            tid = read(f"layers.{i}.ffn.gate.tid2eid")
            for t in range(VOCAB):
                lowest = tid[t][np.argmin(logits[t, tid[t]])]
                self.assertEqual(tables[i][t].tolist(), [e for e in tid[t] if e != lowest])
            self.assertTrue(0 < share[i] < 1 / 6)  # the lowest of six carries less than a sixth

    def test_the_rewrite_moves_the_offsets_pads_the_header_and_keeps_the_other_tensors(self):
        src, dst = self.base / "carried-001.safetensors", self.root / "out.safetensors"
        new = {f"layers.{i}.ffn.gate.tid2eid": np.arange(VOCAB * 4, dtype=np.int64).reshape(VOCAB, 4)
               for i in range(3)}
        with mock.patch.object(tk, "COPY_BYTES", 5):  # stream in many small reads
            tk.rewrite(src, dst, new)
        (s_start, s_table), (o_start, o_table) = mc.header(src), mc.header(dst)
        self.assertEqual(o_start % 8, 0)
        self.assertEqual(list(o_table), list(s_table))
        self.assertEqual(o_table["__metadata__"], {"format": "pt"})
        raw = dst.read_bytes()
        head = raw[8:o_start].decode()
        self.assertEqual(json.loads(head), o_table)  # spaces are valid JSON padding
        end = 0
        for name in tk.data_order(o_table):
            self.assertEqual(o_table[name]["data_offsets"][0], end)  # contiguous, same order
            end = o_table[name]["data_offsets"][1]
        self.assertEqual(len(raw), o_start + end)
        self.assertEqual(o_table["layers.0.ffn.gate.tid2eid"]["shape"], [VOCAB, 4])
        self.assertEqual(mc.read_tensor(dst, "layers.2.ffn.gate.tid2eid").tolist(),
                         new["layers.2.ffn.gate.tid2eid"].tolist())
        for name in s_table:
            if name != "__metadata__" and "tid2eid" not in name:
                self.assertEqual(o_table[name]["shape"], s_table[name]["shape"])
                self.assertTrue(np.array_equal(mc.read_tensor(dst, name), mc.read_tensor(src, name)), name)
        with self.assertRaises(ValueError):
            tk.rewrite(src, self.root / "bad.safetensors", {"big": np.zeros(1000, np.float64)})

    def test_the_config_changes_only_the_top_k(self):
        text, old = tk.edit_config(CONFIG, 5)
        self.assertEqual(old, 6)
        self.assertEqual(text, CONFIG.replace('"num_experts_per_tok": 6', '"num_experts_per_tok": 5'))
        for bad in (6, 0):
            with self.assertRaises(ValueError):
                tk.edit_config(CONFIG, bad)

    def test_a_mask_folder_gives_hardlinks_symlinks_and_a_new_manifest(self):
        record = self.build(self.mask, 5, "top5")
        out = self.root / "top5"
        self.assertEqual(record["links"]["hardlink"], ["carried-002.safetensors"])
        self.assertEqual(os.stat(out / "carried-002.safetensors").st_ino,
                         os.stat(self.mask / "carried-002.safetensors").st_ino)
        for name in ("exl3-layer-000-tp1-rank0.safetensors", "model.safetensors.index.json"):
            self.assertTrue((out / name).is_symlink())
            self.assertEqual(Path(os.readlink(out / name)), (self.base / name).resolve())  # not the mask folder
        self.assertFalse((out / "config.json").is_symlink())
        self.assertEqual(json.loads((out / "config.json").read_text())["num_experts_per_tok"], 5)
        self.assertEqual((self.base / "config.json").read_text(), CONFIG)
        self.assertEqual((out / "MASK.json").read_text(), '{"keep": 10}')
        manifest = json.loads((out / tk.MANIFEST).read_text())
        for item in manifest["files"]:
            self.assertEqual(item["bytes"], (out / item["name"]).stat().st_size)
            self.assertEqual(item["sha256"], hashlib.sha256((out / item["name"]).read_bytes()).hexdigest())
        old = json.loads((self.mask / tk.MANIFEST).read_text())["tensor_bytes"]
        self.assertEqual(manifest["tensor_bytes"], old - 3 * VOCAB * 8)
        self.assertEqual(record["tensor_bytes_change"], -3 * VOCAB * 8)
        topk = json.loads((out / "TOPK.json").read_text())
        self.assertEqual(topk["self_check"]["verify_tp1_manifest"]["tensor_files"], 3)
        self.assertTrue(topk["self_check"]["sha256_matches_manifest"])
        self.assertIn("estimate from the token embedding", topk["rule"])
        self.assertAlmostEqual(topk["dropped_is_least_used"]["random"], 1 / 6)
        tid = mc.read_tensor(out / "carried-001.safetensors", "layers.1.ffn.gate.tid2eid")
        self.assertEqual(tid.shape, (VOCAB, 5))
        with self.assertRaises(FileExistsError):
            self.build(self.mask, 5, "top5")

    def test_the_base_folder_gives_symlinks_only(self):
        record = self.build(self.base, 4, "top4")
        self.assertEqual(record["links"]["hardlink"], [])
        self.assertIn("carried-002.safetensors", record["links"]["symlink"])
        self.assertAlmostEqual(record["dropped_is_least_used"]["random"], 1 / 15)
        self.assertEqual(mc.read_tensor(self.root / "top4/carried-001.safetensors",
                                        "layers.0.ffn.gate.tid2eid").shape, (VOCAB, 4))

    def test_the_self_check_finds_a_changed_byte_and_a_bad_row(self):
        self.build(self.mask, 5, "top5")
        out = self.root / "top5"
        tk.self_check(self.mask, out, 5)
        path = out / "carried-001.safetensors"
        start, table = mc.header(path)
        raw = bytearray(path.read_bytes())
        raw[start + table["big"]["data_offsets"][0]] ^= 1
        path.write_bytes(bytes(raw))
        with self.assertRaisesRegex(ValueError, "big changed its bytes"):
            tk.self_check(self.mask, out, 5)
        src = mc.read_tensor(self.mask / "carried-001.safetensors", "layers.0.ffn.gate.tid2eid")
        with self.assertRaisesRegex(ValueError, "not in the source row"):
            tk.check_rows(src, np.full((VOCAB, 5), 99, np.int64), 5)
        with self.assertRaisesRegex(ValueError, "column order"):
            tk.check_rows(src, src[:, [1, 0, 2, 3, 4]], 5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tk.check_rows(src, src[:, [0, 0, 2, 3, 4]], 5)

    def test_the_least_used_share(self):
        src = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
        order = [5, 4, 3, 2, 1, 0]  # most used first: 0 is the least used
        self.assertEqual(tk.least_used_share(src, np.array([[1, 2], [3, 4]]), order), 0.5)
        self.assertEqual(tk.least_used_share(src, np.array([[2], [3]]), order), 0.5)  # row 2 keeps 3, the least used


if __name__ == "__main__":
    unittest.main()
