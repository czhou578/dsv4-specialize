"""qwen_prune.py on a tiny synthetic checkpoint (plans/PLAN_QWEN_PRUNE.md §5): kept tensors equal their source byte for
byte, the router rows and the draft slices match the kept ids, and the config and the index follow."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import qwen_prune as qp

LAYERS, EXPERTS = 2, 6


def tensor(name, shape, seed, dtype="U8"):
    data = np.random.default_rng(seed).integers(0, 255, size=int(np.prod(shape)) * (2 if dtype == "BF16" else 1), dtype=np.uint8)
    return name, dtype, list(shape), data.tobytes()


def checkpoint(folder):
    """Two files: layer 0 and the draft in the first, layer 1 and the embedding in the second."""
    files = {"model-00001-of-00002.safetensors": [], "model-00002-of-00002.safetensors": []}
    for l, f in zip(range(LAYERS), files.values()):
        p = f"model.language_model.layers.{l}.mlp."
        for e in range(EXPERTS):
            f.append(tensor(f"{p}experts.{e}.down_proj.weight_packed", (4, 3), 100 * l + e))
            f.append(tensor(f"{p}experts.{e}.down_proj.weight_global_scale", (1,), 500 + 100 * l + e))
        f.append(tensor(f"{p}gate.weight", (EXPERTS, 5), 900 + l, "BF16"))
        f.append(tensor(f"{p}shared_expert.down_proj.weight_packed", (4, 3), 950 + l))
    first, second = files.values()
    first += [tensor("mtp.layers.0.mlp.experts.down_proj", (EXPERTS, 4, 2), 1, "BF16"),
              tensor("mtp.layers.0.mlp.experts.gate_up_proj", (EXPERTS, 2, 4), 2, "BF16"),
              tensor("mtp.layers.0.mlp.gate.weight", (EXPERTS, 5), 3, "BF16"), tensor("mtp.fc.weight", (5, 10), 4, "BF16")]
    second.append(tensor("model.language_model.embed_tokens.weight", (7, 5), 5, "BF16"))
    for name, items in files.items():
        qp.write_safetensors(folder / name, items, {"format": "pt"})
    (folder / "config.json").write_text(json.dumps({"model_type": "x", "text_config": {
        "num_hidden_layers": LAYERS, "num_experts": EXPERTS, "num_experts_per_tok": 2}}))
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": {}}))
    (folder / "tokenizer.json").write_text("{}")
    return {t[0]: t for items in files.values() for t in items}


RANK = {"layers": {"0": [{"id": i} for i in (5, 0, 3, 1, 2, 4)], "1": [{"id": i} for i in (2, 4, 1, 0, 3, 5)],
                   "2": [{"id": i} for i in (1, 5, 4, 0, 2, 3)]}}


class PruneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "src"
        self.src.mkdir()
        self.source = checkpoint(self.src)
        self.out = Path(self.tmp.name) / "out"
        self.counts = qp.prune(self.src, self.out, RANK, 3)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, name):
        for f in sorted(self.out.glob("*.safetensors")):
            header, start = qp.read_header(f)
            if name in header:
                h = header[name]
                return h["shape"], f.read_bytes()[start + h["data_offsets"][0]: start + h["data_offsets"][1]]
        return None

    def test_kept_experts_get_new_ids_and_equal_their_source(self):
        kept, dropped, _ = self.counts
        self.assertEqual((kept, dropped), (LAYERS * (3 * 2 + 2) + 4 + 1, LAYERS * 3 * 2))
        for layer, ids in ((0, [0, 3, 5]), (1, [1, 2, 4])):                 # the top 3 of the rank, in the old order
            p = f"model.language_model.layers.{layer}.mlp.experts."
            for new, old in enumerate(ids):
                for part in ("down_proj.weight_packed", "down_proj.weight_global_scale"):
                    self.assertEqual(self.read(f"{p}{new}.{part}")[1], self.source[f"{p}{old}.{part}"][3])
            self.assertIsNone(self.read(f"{p}3.down_proj.weight_packed"))
        self.assertEqual(self.read("model.language_model.embed_tokens.weight")[1],
                         self.source["model.language_model.embed_tokens.weight"][3])

    def test_router_rows_and_draft_slices_follow_the_kept_ids(self):
        rows = lambda name, ids, width: b"".join(self.source[name][3][i * width:(i + 1) * width] for i in ids)
        gate = "model.language_model.layers.0.mlp.gate.weight"
        self.assertEqual(self.read(gate), ([3, 5], rows(gate, [0, 3, 5], 10)))
        for name, width in (("mtp.layers.0.mlp.experts.down_proj", 16), ("mtp.layers.0.mlp.experts.gate_up_proj", 16),
                            ("mtp.layers.0.mlp.gate.weight", 10)):
            shape, data = self.read(name)
            self.assertEqual((shape[0], data), (3, rows(name, [1, 4, 5], width)))    # the rank of layer 2 is the draft
        self.assertEqual(self.read("mtp.fc.weight")[1], self.source["mtp.fc.weight"][3])

    def test_config_index_and_files_follow(self):
        self.assertEqual(json.loads((self.out / "config.json").read_text())["text_config"]["num_experts"], 3)
        index = json.loads((self.out / "model.safetensors.index.json").read_text())
        self.assertEqual(index["weight_map"]["model.language_model.layers.1.mlp.experts.2.down_proj.weight_packed"],
                         "model-00002-of-00002.safetensors")
        self.assertNotIn("model.language_model.layers.1.mlp.experts.3.down_proj.weight_packed", index["weight_map"])
        self.assertTrue((self.out / "tokenizer.json").exists())
        for f in self.out.glob("*.safetensors"):                               # contiguous offsets, an aligned header
            header, start = qp.read_header(f)
            spans = sorted(h["data_offsets"] for k, h in header.items() if k != "__metadata__")
            self.assertTrue(start % 8 == 0 and spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:])))
            self.assertEqual(f.stat().st_size, start + spans[-1][1])

    def test_check_passes_and_finds_a_changed_byte(self):
        self.assertEqual(qp.check(self.src, self.out), self.counts[0])
        f = self.out / "model-00002-of-00002.safetensors"
        raw = bytearray(f.read_bytes())
        raw[-1] ^= 1
        f.write_bytes(raw)
        with self.assertRaises(AssertionError):
            qp.check(self.src, self.out)

    def test_rank_rules(self):
        with self.assertRaises(ValueError):                                    # no draft layer in the rank
            qp.keep_sets({"layers": {k: v for k, v in RANK["layers"].items() if k != "2"}}, 3, LAYERS)
        self.assertEqual(qp.keep_sets(RANK, 2, LAYERS, draft="first")["mtp"], [0, 1])
        out = Path(self.tmp.name) / "keep"
        qp.prune(self.src, out, RANK, 3, draft="keep")                         # the draft stays whole
        header, _ = qp.read_header(out / "model-00001-of-00002.safetensors")
        self.assertEqual(header["mtp.layers.0.mlp.experts.down_proj"]["shape"][0], EXPERTS)
        self.assertEqual(header["model.language_model.layers.0.mlp.gate.weight"]["shape"][0], 3)
        self.assertGreater(qp.check(self.src, out), 0)
        with self.assertRaises(ValueError):
            qp.prune(self.src, Path(self.tmp.name) / "o2", RANK, EXPERTS)      # nothing to prune


if __name__ == "__main__":
    unittest.main()
