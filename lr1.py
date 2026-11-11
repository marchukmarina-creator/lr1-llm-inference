"""LR1: prepare -> check -> run -> analyze -> pack."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata as im
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics as stats
import subprocess
import time
import traceback
import urllib.request
import uuid
import zipfile
from collections import defaultdict, deque
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/experiment.json"
SOURCE_URL = "https://www.gutenberg.org/ebooks/1342.txt.utf-8"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}, line {number}: incomplete record; preserve and inspect") from error
    return rows


def version(package):
    try:
        return im.version(package)
    except im.PackageNotFoundError:
        return "not-installed"


def git(*args):
    return subprocess.check_output(["git", *args], text=True, cwd=ROOT).strip()


def request_json(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def planned(config):
    result = []
    def add(group, length, output, count):
        for i in range(count):
            result.append({"id": f"{group}_{i:03d}", "group": group,
                           "input_target": length, "output_target": output,
                           "concurrency": int(group.split("_")[1]) if group.startswith("load_") else 1})
    for length in config["prompt_lengths"]:
        add(f"exp1_{length}", length, 16, config["repeats"])
    add("cache", 8192, 16, 3)
    for output in config["output_lengths"]:
        add(f"exp2_{output}", 1024, output, config["repeats"])
    for c in config["concurrency"]:
        add(f"load_{c}", 1024, 128, max(16, 4 * c))
    return result


def latency(t0, t1, tn, n):
    if n < 1 or not t0 <= t1 <= tn:
        raise ValueError("Invalid token count or timestamps")
    return {"ttft_s": t1 - t0, "tpot_s": (tn - t1) / (n - 1) if n > 1 else None,
            "e2e_s": tn - t0}


def percentile(values, q):
    """Linear percentile. Failures may be +inf, never silently excluded."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    a, b = math.floor(position), math.ceil(position)
    if a == b:
        return values[a]
    if math.isinf(values[b]):
        return math.inf
    return values[a] + (values[b] - values[a]) * (position - a)


def validate_rows(rows):
    identifiers = [r["id"] for r in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate request IDs")


def clean_revision():
    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError("Commit code, manifest, passport, checks and lock-file before final run:\n" + dirty)
    return git("rev-parse", "HEAD")


def passport(config):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(config["model"], local_files_only=True))
    mc = read_json(snapshot / "config.json")
    log = Path(config["server_log"]).read_text(errors="replace")
    command = Path(config["server_command"]).read_text().strip()
    cap = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)
    memory = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log)
    if not cap or not memory:
        raise RuntimeError("KV cache capacity/memory not found in server log")
    bpt = 2 * mc["num_hidden_layers"] * mc["num_key_value_heads"] * mc["head_dim"] * 2
    capacity = int(cap.group(1).replace(",", ""))
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True).strip()
    index = snapshot / "model.safetensors.index.json"
    weight_bytes = read_json(index)["metadata"].get("total_size") if index.exists() else None
    if weight_bytes is None:
        weight_bytes = sum(p.stat().st_size for p in snapshot.glob("*.safetensors"))
    return {
        "model": config["model"], "revision": snapshot.name,
        "revision_evidence": "Hugging Face cached snapshot resolved locally at preparation",
        "original_startup_revision_pinned": "--revision" in command,
        "weight_format": "FP16 (loaded with --dtype half)", "weight_file_bytes": weight_bytes,
        "server": "vLLM", "server_version": version("vllm"),
        "command": command, "max_model_len": 32768, "enforce_eager": True,
        "prefix_caching": True, "max_num_seqs": 32, "max_num_batched_tokens": 2048,
        "gpu": gpu, "logical_cpu_count": os.cpu_count(),
        "cpu_threads_note": "No CPU thread flag supplied; server defaults apply; environment recorded.",
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "num_hidden_layers": mc["num_hidden_layers"],
        "num_key_value_heads": mc["num_key_value_heads"], "head_dim": mc["head_dim"],
        "kv_bytes_per_token": bpt, "kv_capacity_tokens": capacity,
        "kv_memory_log_gib": float(memory.group(1)),
        "kv_memory_calculated_gib": capacity * bpt / 2**30,
        "kv_32768_gib": 32768 * bpt / 2**30,
        "model_config_hashes": {name: sha(snapshot / name) for name in
            ("config.json", "tokenizer_config.json", "generation_config.json") if (snapshot / name).exists()},
    }


def prepare():
    from transformers import AutoTokenizer
    config = read_json(CONFIG)
    request_json(config["base_url"] + "/models")
    Path("data").mkdir(exist_ok=True)
    ppath = Path("server_passport.json")
    if not ppath.exists():
        write_json(ppath, passport(config))
    pas = read_json(ppath)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=pas["revision"], local_files_only=True)
    textpath = Path("data/pride_and_prejudice.txt")
    if not textpath.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as r:
            textpath.write_bytes(r.read())
    text = textpath.read_text(encoding="utf-8-sig")
    start = re.search(r"\*\*\* START OF .*? \*\*\*", text)
    novel = text[start.end():] if start else text
    ids = tokenizer.encode(novel, add_special_tokens=False)
    if len(ids) < 20000:
        raise ValueError("Source text is unexpectedly short")

    def make_messages(identifier, target, offset=0):
        marker = hashlib.sha256(identifier.encode()).hexdigest()
        instruction = marker + "\nSummarize the following text in one sentence.\n\n"
        count = lambda content: len(tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, return_dict=False))
        take = max(1, target - count(instruction))
        for _ in range(40):
            content = instruction + tokenizer.decode(ids[offset:offset + take])
            observed = count(content)
            if observed == target:
                return [{"role": "user", "content": content}]
            take += target - observed
            if take < 1 or offset + take > len(ids):
                break
        raise ValueError(f"Could not form exact prompt length: {identifier}, {target}")

    entries = planned(config)
    for record in entries:
        offset = int(hashlib.sha256(record["id"].encode()).hexdigest()[:8], 16) % (len(ids) - 2048) if record["group"].startswith("load_") else 0
        identifier = "cache_shared" if record["group"] == "cache" else record["id"]
        record["messages"] = make_messages(identifier, record["input_target"], offset)
    warmups = [
        dict(r, id="warmup_" + r["group"],
             messages=make_messages("warmup_" + r["group"], min(1024, r["input_target"])))
        for r in {e["group"]: e for e in entries}.values()
    ]
    for name, records in (("data/prompts.jsonl", entries), ("data/warmups.jsonl", warmups)):
        with Path(name).open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"datasets": {"pride_and_prejudice": {
        "source": SOURCE_URL, "work": "Pride and Prejudice", "author": "Jane Austen",
        "gutenberg_id": 1342, "source_file": str(textpath), "source_sha256": sha(textpath),
        "selection": "All original bytes retained; text after START tokenized; no records excluded",
        "tokenizer_model": config["model"], "tokenizer_revision": pas["revision"],
        "prompts_file": "data/prompts.jsonl", "prompts_sha256": sha("data/prompts.jsonl"),
        "warmups_sha256": sha("data/warmups.jsonl"),
        "planned_measurements": len(entries), "warmups": len(warmups),
    }}}
    if Path("data/manifest.json").exists() and read_json("data/manifest.json") != manifest:
        raise ValueError("Regenerated data differs from committed manifest; preserve and inspect")
    write_json("data/manifest.json", manifest)
    lock = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
    Path("requirements.lock.txt").write_text(lock, encoding="utf-8")
    for source, destination in (
        (config["server_log"], "data/server_startup.log"),
        (config["server_command"], "data/server_command.txt"),
        ("/content/lr1_checks.json", "data/preliminary_checks.json"),
    ):
        if Path(source).exists():
            shutil.copyfile(source, destination)
    print(f"Prepared {len(entries)} measurements. No experiments executed.")
    print("Commit passport, manifest, server log/command, checks and lock-file before final run.")
    print("KV memory, calculated/logged GiB:", pas["kv_memory_calculated_gib"], pas["kv_memory_log_gib"])


def params(config, record, salt):
    return dict(model=config["model"], messages=record["messages"],
                temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0,
                seed=0, max_tokens=record["output_target"], stop=[],
                stream=True, stream_options={"include_usage": True},
                extra_body={"ignore_eos": True, "top_k": -1, "min_p": 0,
                            "repetition_penalty": 1.0, "cache_salt": salt})


async def timed_request(client, config, record, salt, segment):
    row = {k: v for k, v in record.items() if k != "messages"}
    row.update(status="failed", text="", segment=segment, chunks=[],
               t0=time.perf_counter(), t1=None, tn=None, finish_reason=None,
               prompt_tokens=None, output_tokens=None, ttft_s=None, tpot_s=None, e2e_s=None)
    stream = None
    try:
        stream = await client.chat.completions.create(**params(config, record, salt))
        usage = None
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    row["finish_reason"] = choice.finish_reason
                if choice.delta.content:
                    now = time.perf_counter()
                    row["t1"] = now if row["t1"] is None else row["t1"]
                    row["tn"] = now
                    row["chunks"].append({"time": now, "text": choice.delta.content})
                    row["text"] += choice.delta.content
        if usage is None or row["t1"] is None:
            raise ValueError("Missing usage or nonempty text")
        row.update(prompt_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
                   usage=usage.model_dump())
        row.update(latency(row["t0"], row["t1"], row["tn"], row["output_tokens"]))
        if not record["id"].startswith("warmup_") and row["prompt_tokens"] != record["input_target"]:
            raise ValueError(f"Prompt length mismatch: {row['prompt_tokens']} != {record['input_target']}")
        if row["output_tokens"] != record["output_target"]:
            raise ValueError(f"Output length mismatch: {row['output_tokens']} != {record['output_target']}")
        row["status"] = "completed"
    except Exception as error:
        row.update(error=str(error), traceback=traceback.format_exc())
    finally:
        row["request_end"] = time.perf_counter()
        if stream is not None:
            try:
                await stream.close()
            except Exception as error:
                row["stream_close_error"] = str(error)
    return row


def server_metrics(path):
    try:
        with urllib.request.urlopen(read_json(CONFIG)["base_url"].removesuffix("/v1") + "/metrics", timeout=10) as r:
            text = r.read().decode()
        Path(path).write_text(text, encoding="utf-8")
        values = defaultdict(float)
        for line in text.splitlines():
            for name in ("vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
                if line.startswith(name + "{") or line.startswith(name + " "):
                    values[name] += float(line.split()[-1])
        return dict(values)
    except Exception as error:
        raise RuntimeError(f"Cannot collect required server metrics: {error}") from error


async def group_requests(client, config, records, directory, run_id, segment):
    queue = deque(records)
    async def worker():
        while queue:
            record = queue.popleft()
            salt = run_id + ":" + ("cache_shared" if record["group"] == "cache" else record["id"])
            row = await timed_request(client, config, record, salt, segment)
            append_jsonl(directory / "results.jsonl", row)
            print(f"{row['id']}: {row['status']}, TTFT={row['ttft_s']}, TPOT={row['tpot_s']}", flush=True)
    await asyncio.gather(*(worker() for _ in range(records[0]["concurrency"])))


async def run_group(config, records, warmup, directory, run_id):
    from openai import AsyncOpenAI
    group = records[0]["group"]
    segment = uuid.uuid4().hex
    async with AsyncOpenAI(base_url=config["base_url"], api_key="none",
                           timeout=config["timeout_s"], max_retries=0) as client:
        warm = await timed_request(client, config, warmup, uuid.uuid4().hex, segment)
        append_jsonl(directory / "warmup.jsonl", warm)
        if warm["status"] != "completed":
            raise RuntimeError(f"Warmup failed: {warm.get('error')}")
        if not group.startswith("load_"):
            await group_requests(client, config, records, directory, run_id, segment)
            return
        metricsdir = directory / "server_metrics"
        metricsdir.mkdir(exist_ok=True)
        server_metrics(metricsdir / f"{group}_{segment}_before.txt")
        stop = asyncio.Event()
        async def sample():
            i = 0
            while not stop.is_set():
                values = await asyncio.to_thread(
                    server_metrics, metricsdir / f"{group}_{segment}_{i:04d}.txt")
                append_jsonl(directory / "server_samples.jsonl",
                             {"group": group, "segment": segment, "time": time.perf_counter(), **values})
                i += 1
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        sampler = asyncio.create_task(sample())
        try:
            await group_requests(client, config, records, directory, run_id, segment)
        finally:
            stop.set()
            await sampler
            server_metrics(metricsdir / f"{group}_{segment}_after.txt")


def run(experiment, run_id):
    import mlflow
    config = read_json(CONFIG)
    commit = clean_revision()
    manifest = read_json("data/manifest.json")["datasets"]["pride_and_prejudice"]
    for file, expected in (
        (manifest["source_file"], manifest["source_sha256"]),
        (manifest["prompts_file"], manifest["prompts_sha256"]),
        ("data/warmups.jsonl", manifest["warmups_sha256"]),
    ):
        if sha(file) != expected:
            raise ValueError(f"Input checksum mismatch: {file}")
    plan = jsonl("data/prompts.jsonl")
    validate_rows(plan)
    warmups = {r["group"]: r for r in jsonl("data/warmups.jsonl")}
    directory = Path("runs") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "commit": commit, "config_sha256": sha(CONFIG), "manifest_sha256": sha("data/manifest.json"),
        "passport_sha256": sha("server_passport.json"), "python": platform.python_version(),
        "versions": {p: version(p) for p in ("openai", "vllm", "torch", "mlflow", "transformers", "matplotlib")},
    }
    mpath = directory / "metadata.json"
    metadata = read_json(mpath) if mpath.exists() else {
        "run_id": run_id, "experiment": "lr1", "started_at": utc(),
        "fingerprint": fingerprint, "planned_measurements": len(plan),
    }
    if metadata["fingerprint"] != fingerprint:
        raise ValueError("Code/config/data/environment changed: new series required; preserve old records")
    existing = jsonl(directory / "results.jsonl")
    validate_rows(existing)
    completed_ids = {r["id"] for r in existing}  # Includes failures; never silently retry them.
    selected = [r for r in plan if (experiment == "all" or r["group"].split("_")[0] == experiment)
                and r["id"] not in completed_ids]
    if not selected:
        print("No missing requests. Saved successes and failures were preserved.")
        return
    write_json(directory / "config.yaml", config)
    for src, dst in (("server_passport.json", "server_passport.json"),
                     ("data/manifest.json", "manifest.json"),
                     ("data/prompts.jsonl", "requests.jsonl"),
                     ("requirements.lock.txt", "requirements.lock.txt")):
        shutil.copyfile(src, directory / dst)
    mlflow.set_tracking_uri("sqlite:///" + str((ROOT / "mlflow.db").resolve()))
    mlflow.set_experiment("lr1")
    context = mlflow.start_run(run_id=metadata["mlflow_run_id"]) if metadata.get("mlflow_run_id") else mlflow.start_run(run_name=run_id)
    metadata.update(status="running", last_resumed_at=utc())
    with context as parent:
        metadata["mlflow_run_id"] = parent.info.run_id
        write_json(mpath, metadata)
        mlflow.set_tags({"work": "lr1", "series": "final", "commit": commit, "model": config["model"]})
        mlflow.log_params({**config, "temperature": 0, "ignore_eos": True})
        groups = defaultdict(list)
        for r in selected:
            groups[r["group"]].append(r)
        try:
            for name, records in groups.items():
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.set_tags({"work": "lr1", "series": name, "commit": commit})
                    mlflow.log_params({k: records[0][k] for k in ("input_target", "output_target", "concurrency")})
                    asyncio.run(run_group(config, records, warmups[name], directory, run_id))
                    rows = [r for r in jsonl(directory / "results.jsonl") if r["group"] == name]
                    for i, row in enumerate(rows):
                        mlflow.log_metric("success", int(row["status"] == "completed"), step=i)
                        if row["status"] == "completed":
                            mlflow.log_metrics({k: row[k] for k in ("ttft_s", "tpot_s", "e2e_s") if row[k] is not None}, step=i)
                    groupfile = directory / f"{name}.jsonl"
                    groupfile.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
                    mlflow.log_artifact(str(groupfile))
            metadata.update(status="completed" if len(jsonl(directory / "results.jsonl")) == len(plan) else "partial",
                            finished_at=utc())
        except BaseException as error:
            metadata.update(status="failed", error=repr(error), traceback=traceback.format_exc(), finished_at=utc())
            raise
        finally:
            all_rows = jsonl(directory / "results.jsonl")
            metadata["saved_measurements"] = len(all_rows)
            metadata["failed_measurements"] = sum(r["status"] != "completed" for r in all_rows)
            write_json(mpath, metadata)
            mlflow.log_artifacts(str(directory))
    print("Saved:", directory, metadata["status"])


def summarize(rows, expected, config):
    validate_rows(rows)
    if len(rows) > expected:
        raise ValueError("More results than planned")
    ok = [r for r in rows if r["status"] == "completed"]
    segments = defaultdict(list)
    for r in rows:
        segments[r["segment"]].append(r)
    elapsed = sum(max(r["tn"] if r["status"] == "completed" else r["request_end"] for r in part)
                  - min(r["t0"] for r in part) for part in segments.values())
    good = sum(r["ttft_s"] <= config["ttft_limit_s"] and r["tpot_s"] is not None
               and r["tpot_s"] <= config["tpot_limit_s"] for r in ok)
    values = [r["ttft_s"] if r["status"] == "completed" else math.inf for r in rows]
    values.extend([math.inf] * (expected - len(rows)))
    p90 = percentile(values, 0.9)
    times = [r["ttft_s"] for r in ok]
    tpots = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    result = {
        "planned": expected, "saved": len(rows), "completed": len(ok),
        "failed": sum(r["status"] != "completed" for r in rows), "missing": expected - len(rows),
        "segments": len(segments), "active_time_s": elapsed,
        "ttft_median_s": stats.median(times) if times else None,
        "ttft_min_s": min(times) if times else None, "ttft_max_s": max(times) if times else None,
        "ttft_p90_s": p90 if p90 is not None and math.isfinite(p90) else None,
        "ttft_p90_unbounded": p90 is not None and not math.isfinite(p90),
        "tpot_median_s": stats.median(tpots) if tpots else None,
        "e2e_median_s": stats.median(r["e2e_s"] for r in ok) if ok else None,
        "e2e_min_s": min(r["e2e_s"] for r in ok) if ok else None,
        "e2e_max_s": max(r["e2e_s"] for r in ok) if ok else None,
        "output_tokens_per_s": sum(r.get("output_tokens") or 0 for r in rows) / elapsed if elapsed > 0 else None,
        "completed_requests_per_s": len(ok) / elapsed if elapsed > 0 else None,
        "good_requests": good, "goodput_requests_per_s": good / elapsed if elapsed > 0 else None,
    }
    result["passes_limits"] = (len(ok) == expected and p90 is not None
        and p90 <= config["ttft_limit_s"] and bool(tpots)
        and stats.median(tpots) <= config["tpot_limit_s"])
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_id):
    import mlflow
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path("runs") / run_id
    metadata = read_json(directory / "metadata.json")
    config = read_json(directory / "config.yaml")
    rows = jsonl(directory / "results.jsonl")
    validate_rows(rows)
    plan = jsonl(directory / "requests.jsonl")
    expected, actual = defaultdict(list), defaultdict(list)
    planned_ids = {r["id"] for r in plan}
    for r in plan:
        expected[r["group"]].append(r)
    for r in rows:
        if r["id"] not in planned_ids:
            raise ValueError("Unexpected result id")
        if r["status"] == "completed":
            if r["output_tokens"] != r["output_target"] or r["prompt_tokens"] != r["input_target"]:
                raise ValueError("Successful result has wrong token counts")
            r.update(latency(r["t0"], r["t1"], r["tn"], r["output_tokens"]))
        actual[r["group"]].append(r)
    summaries = []
    samples = jsonl(directory / "server_samples.jsonl")
    for group, requests in expected.items():
        data = summarize(actual[group], len(requests), config)
        data.update(group=group, input_tokens=requests[0]["input_target"],
                    output_tokens=requests[0]["output_target"], concurrency=requests[0]["concurrency"])
        observations = [s for s in samples if s["group"] == group]
        data["max_waiting"] = max((s.get("vllm:num_requests_waiting", 0) for s in observations), default=None)
        data["max_kv_usage_fraction"] = max((s.get("vllm:kv_cache_usage_perc", s.get("vllm:gpu_cache_usage_perc", 0)) for s in observations), default=None)
        summaries.append(data)
    reports, figures = Path("reports"), Path("reports/figures")
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(reports / "conditions.csv", summaries)
    load = [s for s in summaries if s["group"].startswith("load_")]
    exp1 = [s for s in summaries if s["group"].startswith("exp1_")]
    exp2 = [s for s in summaries if s["group"].startswith("exp2_")]
    write_csv(reports / "load.csv", load)
    for s in exp1:
        s["prefill_tokens_per_s"] = s["input_tokens"] / s["ttft_median_s"] if s["ttft_median_s"] else None
    write_csv(reports / "prefill.csv", exp1)
    write_csv(reports / "decode.csv", exp2)
    complete2 = [s for s in exp2 if s["e2e_median_s"] is not None]
    fit = None
    if len(complete2) >= 2:
        xs = [s["output_tokens"] - 1 for s in complete2]
        ys = [s["e2e_median_s"] for s in complete2]
        xm, ym = stats.mean(xs), stats.mean(ys)
        slope = sum((x-xm)*(y-ym) for x,y in zip(xs,ys)) / sum((x-xm)**2 for x in xs)
        fit = {"formula": "e2e = intercept + slope * (N - 1)",
               "intercept_s": ym - slope*xm, "slope_s_per_token": slope}
    cache = sorted(actual["cache"], key=lambda r: r["id"])
    speeds = None
    if len(cache) == 3 and all(r["status"] == "completed" for r in cache):
        speeds = [cache[0]["ttft_s"] / r["ttft_s"] for r in cache[1:]]

    def curve(items, xkey, ykey, name, xlabel, ylabel, low=None, high=None):
        items = [s for s in items if s[ykey] is not None]
        if not items:
            return
        x, y = [s[xkey] for s in items], [s[ykey] for s in items]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        errors = [[s[ykey]-s[low] for s in items], [s[high]-s[ykey] for s in items]] if low else None
        ax.errorbar(x, y, yerr=errors, fmt="o-", capsize=4)
        if name == "decode.png" and fit:
            ax.plot(x, [fit["intercept_s"] + fit["slope_s_per_token"]*(n-1) for n in x],
                    "--", label="Fit: intercept + slope*(N-1)")
            ax.legend()
        if name == "throughput_latency.png":
            for s in items:
                ax.annotate(f"c={s['concurrency']}", (s[xkey], s[ykey]))
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=160)
        plt.close(fig)
    curve(exp1, "input_tokens", "ttft_median_s", "prefill.png",
          "Input tokens (chat template included)", "TTFT, s (median and min-max)", "ttft_min_s", "ttft_max_s")
    curve(exp2, "output_tokens", "e2e_median_s", "decode.png",
          "Output tokens", "End-to-end, s (median and min-max)", "e2e_min_s", "e2e_max_s")
    curve(load, "output_tokens_per_s", "ttft_p90_s", "throughput_latency.png",
          "Output tokens / second", "p90 TTFT, s (failed requests = infinity)")
    passed = [s["concurrency"] for s in load if s["passes_limits"]]
    pas = read_json(directory / "server_passport.json")
    summary = {
        "run_id": run_id, "metadata_commit": metadata["fingerprint"]["commit"],
        "planned": len(plan), "saved": len(rows), "failed": sum(r["status"] != "completed" for r in rows),
        "max_concurrency_meeting_limits": max(passed) if passed else None,
        "decode_fit": fit, "prefix_cache_speedup_repeat2_repeat3": speeds,
        "kv_capacity_short_requests_upper_bound": pas["kv_capacity_tokens"] // (1024+128),
        "kv_8_users_4000_tokens_gib": 8 * 4000 * pas["kv_bytes_per_token"] / 2**30,
        "conditions": summaries,
    }
    write_json(reports / "summary.json", summary)
    write_json(directory / "metrics.json", {k:v for k,v in summary.items() if k != "conditions"})

    def table(items, keys):
        out = "|" + "|".join(keys) + "|\n|" + "|".join("---" for _ in keys) + "|\n"
        for s in items:
            out += "|" + "|".join(f"{s[k]:.4f}" if isinstance(s[k],float) else str(s[k]) for k in keys) + "|\n"
        return out
    report = (
        "# ЛР1. Инференс и развёртывание модели\n\n"
        "## Вопрос и прогноз\n\nИз чего складывается время ответа и как оно меняется под нагрузкой?\n\n"
        + Path("configs/prediction.md").read_text(encoding="utf-8")
        + "\n\n## Стенд и данные\n\n"
        + f"Серия {run_id}, коммит {summary['metadata_commit']}. Модель {pas['model']}, "
          f"ревизия {pas['revision']}, {pas['weight_format']}, vLLM {pas['server_version']}. "
          f"Оборудование: {pas['gpu']}. Команда и CPU-параметры: server_passport.json. "
          f"KV: {pas['kv_bytes_per_token']} байт/токен, {pas['kv_capacity_tokens']} токенов, "
          f"{pas['kv_memory_calculated_gib']:.3f} GiB; журнал: {pas['kv_memory_log_gib']} GiB. "
          f"Кеш 32768 токенов: {pas['kv_32768_gib']:.2f} GiB; веса: около {pas['weight_file_bytes']/1e9:.2f} GB. "
          "Текст: Pride and Prejudice, Project Gutenberg №1342; хеши в манифесте. "
          "Длины после шаблона диалога и длины ответов проверены по usage. "
          "Temperature=0, остальные параметры явные. Прогрев вне статистики. "
          "Уникальная метка в начале и cache_salt исключают повторный prefill из кеша; "
          "в опыте кеширования соль и текст общие для трёх повторов.\n\n"
        + "## Эксперимент 1: длина запроса\n\n"
          "Пять длин, по 5 запросов на длину, ответ 16 токенов. "
          "Задержки в таблицах относятся к успешным запросам; показаны все знаменатели. "
          "Скорость prefill оценена как входные токены / TTFT (включая клиентские затраты).\n\n"
        + table(exp1, ["input_tokens", "completed", "planned", "ttft_median_s",
                       "ttft_min_s", "ttft_max_s", "prefill_tokens_per_s"])
        + "\n![Prefill](figures/prefill.png)\n\n"
        + f"Ускорение повторов 2 и 3 относительно первого при prefix caching: {speeds}. "
          "Смысл для агента: описания инструментов и неизменная история могут переиспользовать KV.\n\n"
        + table([{"repeat":i+1, "ttft_s":r["ttft_s"], "status":r["status"]} for i,r in enumerate(cache)],
                ["repeat", "ttft_s", "status"])
        + "\n## Эксперимент 2: длина ответа\n\n"
          "Запрос 1024 токена; ответы 16/64/256/1024, по 5 запросов. "
          f"Аппроксимация по медианам: {fit}. N−1 используется, поскольку первый токен учтён в TTFT. "
          "Оценка нижней границы TPOT: 8/300 ≈ 0.027 с; реальное значение включает "
          "чтение кеша, вычисления, планировщик и клиентский streaming.\n\n"
        + table(exp2, ["output_tokens", "completed", "planned", "e2e_median_s", "tpot_median_s", "ttft_median_s"])
        + "\n![Decode](figures/decode.png)\n\n"
        + "## Эксперимент 3: нагрузка\n\n"
          "Замкнутый цикл запрос-ответ у каждого клиента: 1024 входных, 128 выходных. "
          "На уровне max(16,4c) запросов. Пределы: p90 TTFT ≤2 с, медиана TPOT ≤0.1 с. "
          "Goodput: количество запросов с обоими индивидуальными пределами / время уровня.\n\n"
        + table(load, ["concurrency", "completed", "planned", "ttft_p90_s", "tpot_median_s",
                       "output_tokens_per_s", "goodput_requests_per_s", "passes_limits"])
        + "\n![Load](figures/throughput_latency.png)\n\n"
        + f"Наибольшая прошедшая проверенная нагрузка: {summary['max_concurrency_meeting_limits']}. "
          f"Теоретическая верхняя оценка KV для запросов 1024+128: {summary['kv_capacity_short_requests_upper_bound']}. "
          "Это ограничение памяти, а не обещание скорости; max_num_seqs=32 также ограничивает батч. "
          "Очередь и занятость кеша измерялись до/после и во время нагрузки. "
          "Их максимумы приведены в load.csv. Если очередь не наблюдалась, нельзя "
          "объяснять задержки заполнением кеша без наблюдения.\n\n"
        + "## Выводы, ограничения и проверка\n\n"
        + f"Сохранено {len(rows)} из {len(plan)} измерений; неуспехов {summary['failed']}. "
          "Неполная серия не является окончательным результатом. Медианы рассчитаны по успешным; "
          "p90 TTFT включает неуспехи как бесконечность. None и ttft_p90_unbounded в CSV "
          "означают бесконечную оценку либо отсутствие данных; на графике такие точки отсутствуют. "
          "Уровень проходит пределы только при отсутствии ошибок. "
          "При segments>1 уровень возобновлялся: throughput рассчитан по сумме активных "
          "интервалов, что ограничивает сопоставимость с непрерывными уровнями. "
          "Измеренная допустимая нагрузка определяет параллельность ЛР2–ЛР8; документы и "
          "описания инструментов увеличат prefill и расход KV. "
          "Вывод относится к T4, данной модели, eager-режиму и данному профилю запросов.\n\n"
          "Код подготовлен с ИИ. Проверки: test_lr1.py, check (формулы, токены, повтор текста), "
          "хеши, чистый Git, пересчёт метрик из results.jsonl. Итоговый текст интерпретации "
          "дополняется после просмотра реальных графиков, без изменения измерений.\n\n"
          "Источники: [задание](https://github.com/dm780009/intsys/blob/main/LR1_llm_inference_serving.pdf), "
          "[модель](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), "
          "[vLLM](https://docs.vllm.ai/en/v0.33.0/), [Gutenberg](https://www.gutenberg.org/ebooks/1342).\n"
    )
    (reports / "report.md").write_text(report, encoding="utf-8")
    # The independently audited renderer adds the interpretation and preserves
    # the measured source commit rather than claiming an analysis reruns the GPU.
    from finalize_report import report_content, markdown
    report_pages, audit_result = report_content(ROOT)
    (reports / "report.md").write_text(markdown(report_pages), encoding="utf-8")
    write_json(reports / "audit.json", audit_result)
    mlflow.set_tracking_uri("sqlite:///" + str((ROOT / "mlflow.db").resolve()))
    mlflow.set_experiment("lr1")
    with mlflow.start_run(run_id=metadata["mlflow_run_id"]):
        with mlflow.start_run(run_name="analysis", nested=True):
            mlflow.set_tags({"work": "lr1", "series": "analysis", "commit": git("rev-parse", "HEAD"), "source_run": run_id})
            if summary["max_concurrency_meeting_limits"] is not None:
                mlflow.log_metric("max_concurrency_meeting_limits", summary["max_concurrency_meeting_limits"])
            mlflow.log_artifacts(str(reports), artifact_path="reports")
            mlflow.log_artifact(str(directory / "metrics.json"))
    print(json.dumps({k:v for k,v in summary.items() if k != "conditions"}, ensure_ascii=False, indent=2))
    print("Tables, figures, report generated and saved to MLflow.")


def check():
    from openai import AsyncOpenAI
    config = read_json(CONFIG)
    arithmetic = latency(0, .4, 2.4, 101)
    assert math.isclose(arithmetic["ttft_s"], .4)
    assert math.isclose(arithmetic["tpot_s"], .02)
    record = {"id": "warmup_check", "group": "check", "input_target": 0,
              "output_target": 64, "concurrency": 1,
              "messages": [{"role": "user", "content": "Explain in English why the sky is blue."}]}
    async def twice():
        async with AsyncOpenAI(base_url=config["base_url"], api_key="none",
                               timeout=config["timeout_s"], max_retries=0) as client:
            return [await timed_request(client, config, record, "check_same", "check") for _ in range(2)]
    records = asyncio.run(twice())
    if not all(r["status"] == "completed" for r in records):
        raise RuntimeError(json.dumps(records, ensure_ascii=False, indent=2))
    pas = read_json("server_passport.json")
    probe = dict(jsonl("data/prompts.jsonl")[0], id="check_prompt_256")
    async def check_prompt():
        async with AsyncOpenAI(base_url=config["base_url"], api_key="none",
                               timeout=config["timeout_s"], max_retries=0) as client:
            return await timed_request(client, config, probe, uuid.uuid4().hex, "check")
    probe_result = asyncio.run(check_prompt())
    if probe_result["status"] != "completed":
        raise RuntimeError("Prepared prompt failed server length check: " + json.dumps(probe_result))
    result = {"arithmetic": arithmetic, "output_tokens": [r["output_tokens"] for r in records],
              "texts_match": records[0]["text"] == records[1]["text"],
              "kv_memory_difference_gib": pas["kv_memory_calculated_gib"]-pas["kv_memory_log_gib"],
              "records": records, "prepared_prompt_probe": probe_result}
    write_json("data/checks.json", result)
    print("CHECKS OK; tokens:", result["output_tokens"], "; texts match:", result["texts_match"])
    print("KV calculation/log difference GiB:", result["kv_memory_difference_gib"])


def pack(run_id):
    bundle = Path("runs") / run_id / "source_history.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), "--all"], check=True)
    dest = Path(f"lr1_{run_id}_artifacts.zip")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for root in ("runs", "mlruns", "mlartifacts", "reports", "data"):
            if Path(root).exists():
                for path in Path(root).rglob("*"):
                    if path.is_file():
                        z.write(path, path.as_posix())
        for name in ("mlflow.db", "endpoint.yaml", "server_passport.json", "requirements.lock.txt"):
            if Path(name).exists():
                z.write(name, name)
    print("BACKUP:", dest.resolve())


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "check", "run", "analyze", "pack"])
    parser.add_argument("--experiment", choices=["all", "exp1", "cache", "exp2", "load"], default="all")
    parser.add_argument("--run-id", default="lr1_final")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id must contain only letters, digits, _ or -")
    if args.action == "prepare":
        prepare()
    elif args.action == "check":
        check()
    elif args.action == "run":
        run(args.experiment, args.run_id)
    elif args.action == "analyze":
        analyze(args.run_id)
    elif args.action == "pack":
        pack(args.run_id)


if __name__ == "__main__":
    main()
