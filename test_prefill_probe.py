"""prefill_probe.py: cold prompts, the warm turn and the smoke requests, against a fake server."""

import contextlib
import io
import json
import re
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import prefill_probe


class FakeServer:
    """10 template tokens plus one token per filler word. A reply gives the prompt's code unless `lose_code`.
    With `cache`, a prompt that extends an earlier one reports that earlier prompt as cached."""

    def __init__(self, cache=True, lose_code=False, json_reply='{"ticker": "NVDA", "price": 123.4}',
                 tool_calls=1, healthy=True):
        self.cache, self.lose_code, self.json_reply = cache, lose_code, json_reply
        self.tool_calls, self.up, self.seen = tool_calls, healthy, []

    @staticmethod
    def tokens(text):
        return 10 + text.count(" x") + text.count(" y")

    def get(self, base, path, timeout=None):
        return {"data": [{"id": "fake-model"}]}

    def post(self, base, path, body, timeout=None):
        if path == "/tokenize":
            return {"count": self.tokens(body["messages"][0]["content"])}
        if "response_format" in body:
            return {"choices": [{"message": {"content": self.json_reply}}]}
        calls = [{"function": {"name": "get_quote", "arguments": '{"ticker": "NVDA"}'}}] * self.tool_calls
        return {"choices": [{"message": {"content": "", "tool_calls": calls}}]}

    def stream(self, base, body, timeout=None):
        text = body["messages"][0]["content"]
        key = text.split("\nWhat is")[0]
        cached = max([self.tokens(k) for k in self.seen if self.cache and key.startswith(k)], default=0)
        self.seen.append(key)
        code = "garbage" if self.lose_code else re.search(r"QX-\d{5}", text).group()
        yield 0.5, {"choices": [{"delta": {"role": "assistant"}}]}
        yield 2.0, {"choices": [{"delta": {"content": code}}]}
        yield 2.1, {"choices": [], "usage": {"prompt_tokens": self.tokens(text),
                                             "prompt_tokens_details": {"cached_tokens": cached}}}

    def run(self, *args):
        patches = {"get": self.get, "post": self.post, "stream": self.stream,
                   "healthy": lambda base: self.up, "mem_available_mib": lambda: 5000}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.multiple(prefill_probe, **patches), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            out = Path(tmp) / "probe.json"
            code = prefill_probe.main(["--base-url", "http://fake", "--out", str(out), *args])
            return code, json.loads(out.read_text()), err.getvalue()


class ProbeTest(unittest.TestCase):
    def test_a_cold_prompt_gets_its_size_its_first_token_time_and_the_code(self):
        code, result, err = FakeServer().run("--cold", "1000", "4000")
        self.assertEqual(code, 0)
        first = result["cold"][0]
        self.assertEqual((first["prompt_tokens"], first["ttft_s"], first["prefill_tok_s"]), (1000, 2.0, 500.0))
        self.assertTrue(first["recall"])
        self.assertEqual(first["min_avail_mib"], 5000)
        self.assertEqual(result["cold"][1]["prompt_tokens"], 4000)
        self.assertTrue(err.startswith("PASS; cold 1000: 1000 tok, ttft 2.0 s"))

    def test_a_lost_code_or_a_server_that_goes_down_fails(self):
        code, result, _ = FakeServer(lose_code=True).run("--cold", "1000")
        self.assertEqual(code, 1)
        self.assertFalse(result["cold"][0]["recall"])
        code, result, _ = FakeServer(healthy=False).run("--cold", "1000")
        self.assertEqual(code, 1)
        self.assertTrue(result["cold"][0]["recall"])
        self.assertFalse(result["cold"][0]["healthy"])

    def test_the_warm_turn_must_hit_the_prefix_cache(self):
        code, result, _ = FakeServer().run("--warm", "2000", "300")
        self.assertEqual(code, 0)
        warm = result["warm"]
        self.assertEqual(warm["second"]["prompt_tokens"], 2300)
        self.assertEqual(warm["second"]["cached_tokens"], 2000)
        self.assertEqual(warm["second"]["prefill_tok_s"], 150.0)   # only the 300 new tokens
        code, result, _ = FakeServer(cache=False).run("--warm", "2000", "300")
        self.assertEqual(code, 1)
        self.assertFalse(result["warm"]["hit"])

    def test_smoke_checks_the_json_schema_reply_and_both_tool_choices(self):
        code, result, err = FakeServer().run("--smoke")
        self.assertEqual(code, 0)
        self.assertIn("smoke: json_schema pass tool_auto pass tool_required pass", err)
        code, result, _ = FakeServer(tool_calls=0, json_reply="not json").run("--smoke")
        smoke = result["smoke"]
        self.assertEqual(code, 1)
        self.assertTrue(smoke["tool_auto"]["pass"])        # auto may answer without a call
        self.assertFalse(smoke["tool_required"]["pass"])
        self.assertIn("JSONDecodeError", smoke["json_schema"]["error"])

    def test_an_http_error_is_recorded_and_fails_the_check(self):
        server = FakeServer()
        server.post = mock.Mock(side_effect=urllib.error.URLError("refused"))
        code, result, _ = server.run("--smoke")
        self.assertEqual(code, 1)
        self.assertIn("refused", result["smoke"]["tool_auto"]["error"])


if __name__ == "__main__":
    unittest.main()
