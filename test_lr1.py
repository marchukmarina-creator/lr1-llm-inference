"""CPU-only tests of timing, denominator, streaming and cache isolation."""
import asyncio
import math
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import lr1


class Usage:
    prompt_tokens = 256
    completion_tokens = 101
    def model_dump(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


def chunk(text=None, usage=None, finish=None):
    return types.SimpleNamespace(
        usage=usage,
        choices=[] if usage else [types.SimpleNamespace(
            delta=types.SimpleNamespace(content=text), finish_reason=finish)])


class Stream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration
    async def close(self):
        pass


class Tests(unittest.TestCase):
    config = {"model": "test", "ttft_limit_s": 2, "tpot_limit_s": .1}

    def test_reference_arithmetic(self):
        got = lr1.latency(0, .4, 2.4, 101)
        self.assertAlmostEqual(got["ttft_s"], .4)
        self.assertAlmostEqual(got["tpot_s"], .02)
        self.assertAlmostEqual(got["e2e_s"], 2.4)

    def test_one_token_has_no_tpot(self):
        self.assertIsNone(lr1.latency(0, .4, .4, 1)["tpot_s"])

    def test_invalid_times(self):
        with self.assertRaises(ValueError):
            lr1.latency(0, 2, 1, 10)

    def test_plan_and_counts(self):
        plan = lr1.planned(lr1.read_json(lr1.CONFIG))
        self.assertEqual(len(plan), 320)
        lr1.validate_rows(plan)
        self.assertEqual(sum(p["group"].startswith("load_") for p in plan), 272)
        self.assertEqual(sum(p["group"].startswith("exp1_") for p in plan), 25)
        self.assertEqual(sum(p["group"].startswith("exp2_") for p in plan), 20)
        self.assertEqual(sum(p["group"] == "cache" for p in plan), 3)

    def test_quantile_and_failures(self):
        self.assertAlmostEqual(lr1.percentile([1,2,3,4,5], .9), 4.6)
        self.assertTrue(math.isinf(lr1.percentile([1,2,3,4,math.inf], .9)))
        self.assertIsNone(lr1.percentile([], .9))

    def test_empty_summary_is_not_success(self):
        s = lr1.summarize([], 16, self.config)
        self.assertEqual(s["missing"], 16)
        self.assertFalse(s["passes_limits"])
        self.assertTrue(s["ttft_p90_unbounded"])

    def test_denominator_throughput_goodput(self):
        good = dict(id="a", status="completed", segment="s", t0=0, tn=2,
                    request_end=2.1, ttft_s=.5, tpot_s=.05, e2e_s=2, output_tokens=64)
        slow = dict(good, id="b", t0=.1, tn=3, request_end=3.1, ttft_s=2.1)
        bad = dict(id="c", status="failed", segment="s", t0=.2, tn=None,
                   request_end=4, output_tokens=None)
        s = lr1.summarize([good, slow, bad], 3, self.config)
        self.assertEqual(s["planned"], 3)
        self.assertEqual(s["completed"], 2)
        self.assertEqual(s["failed"], 1)
        self.assertEqual(s["output_tokens_per_s"], 32)
        self.assertEqual(s["goodput_requests_per_s"], .25)
        self.assertFalse(s["passes_limits"])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            lr1.validate_rows([{"id":"a"}, {"id":"a"}])

    def test_resume_segments_exclude_idle_gap(self):
        a = dict(id="a", status="completed", segment="1", t0=0, tn=2,
                 request_end=2, ttft_s=.2, tpot_s=.02, e2e_s=2, output_tokens=100)
        b = dict(a, id="b", segment="2", t0=100, tn=102)
        s = lr1.summarize([a,b], 2, self.config)
        self.assertEqual(s["segments"], 2)
        self.assertEqual(s["active_time_s"], 4)
        self.assertEqual(s["output_tokens_per_s"], 50)

    def test_sampling_parameters_explicit(self):
        p = lr1.params(self.config, {"messages":[], "output_target":64}, "unique")
        self.assertEqual(p["temperature"], 0)
        self.assertTrue(p["stream_options"]["include_usage"])
        self.assertTrue(p["extra_body"]["ignore_eos"])
        self.assertEqual(p["extra_body"]["cache_salt"], "unique")

    def test_stream_ignores_role_and_usage_chunks_in_timing(self):
        async def create(**kwargs):
            return Stream([chunk(), chunk("a"), chunk("b"), chunk("c"),
                           chunk(finish="length"), chunk(usage=Usage())])
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)))
        record = {"id":"x", "group":"exp1", "input_target":256, "output_target":101,
                  "messages":[], "concurrency":1}
        with patch.object(lr1.time, "perf_counter", side_effect=[0,.4,1.2,2.4,2.5]):
            # asyncio also reads its clock, so supply a private event loop clock.
            loop = asyncio.new_event_loop()
            loop.time = lambda: 0
            try:
                row = loop.run_until_complete(lr1.timed_request(client, self.config, record, "salt", "s"))
            finally:
                loop.close()
        self.assertEqual(row["status"], "completed")
        self.assertEqual(len(row["chunks"]), 3)
        self.assertEqual(row["output_tokens"], 101)
        self.assertAlmostEqual(row["tpot_s"], .02)
        self.assertAlmostEqual(row["ttft_s"], .4)

    def test_error_is_returned_as_failed_record(self):
        async def create(**kwargs):
            raise RuntimeError("test error")
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)))
        record = {"id":"x", "group":"exp1", "input_target":256, "output_target":16,
                  "messages":[], "concurrency":1}
        row = asyncio.run(lr1.timed_request(client, self.config, record, "salt", "s"))
        self.assertEqual(row["status"], "failed")
        self.assertIn("test error", row["error"])
        self.assertGreaterEqual(row["request_end"], row["t0"])


if __name__ == "__main__":
    unittest.main()
