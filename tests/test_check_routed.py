"""check_routed.py on synthetic routing: 4 layers (3 hash layers), top-3."""

import base64
import io
import unittest

import numpy as np

import check_routed as cr

TABLE = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 6, 7], [2, 3, 4]])
TOKENS = [0, 1, 2, 3]  # 3 routing rows: the last token has not gone through the model


def routing(learned):
    hash_rows = np.stack([TABLE[t][::-1] for t in TOKENS[:-1]])  # any order of the same experts
    return np.stack([hash_rows, hash_rows, hash_rows, np.array(learned)], axis=1).astype(np.uint8)


class CheckTest(unittest.TestCase):
    tables = {i: TABLE for i in range(3)}

    def test_a_clean_reply_passes(self):
        buf = io.BytesIO()
        np.save(buf, routing([[0, 1, 3], [4, 5, 7], [0, 3, 5]]))
        routed = cr.decode(base64.b64encode(buf.getvalue()).decode())
        result = cr.check(routed, TOKENS, {3: [2, 6]}, self.tables)
        self.assertEqual(result[3], {"distinct": 6})  # no problem keys: only the count
        self.assertEqual(result[0], {"distinct": 8})

    def test_dropped_ids_zero_rows_and_wrong_hash_rows_fail(self):
        routed = routing([[0, 2, 3], [0, 0, 0], [6, 1, 5]])
        result = cr.check(routed, TOKENS, {3: [2, 6]}, self.tables)
        self.assertEqual((result[3]["dropped"], result[3]["uncaptured"]), (2, 1))
        routed[1, 0, :] = [3, 4, 6]  # token 1 maps to 3, 4, 5 in layer 0
        self.assertEqual(cr.check(routed, TOKENS, {}, self.tables)[0]["hash_mismatch"], 1)
        with self.assertRaises(ValueError):
            cr.check(routed, TOKENS + [9], {}, self.tables)


if __name__ == "__main__":
    unittest.main()
