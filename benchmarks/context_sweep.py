import argparse, csv, json, re, shlex
import os, sys, time, subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from transformers import AutoTokenizer


@dataclass
class RunConfig:
    model_key: str
    script_path: str
    model_path: str
    use_chat_template: bool
    offload_dir: Optional[str]


def parse_bins(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("At least one context bin is required.")
    return sorted(set(values))


def parse_profile_bins(text: str, bins: List[int]) -> Set[int]:
    raw = text.strip().lower()
    if raw in ("", "none"):
        return set()
    if raw == "all":
        return set(bins)
    if raw == "minmidmax":
        if len(bins) == 1:
            return {bins[0]}
        mid = bins[len(bins) // 2]
        return {bins[0], mid, bins[-1]}
    return set(parse_bins(text))


def format_prompt(tokenizer, prompt_text: str, use_chat_template: bool) -> str:
    if use_chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if not prompt_text.strip():
        if tokenizer.eos_token:
            return tokenizer.eos_token
        return " "
    return prompt_text


def prompt_len(tokenizer, prompt_text: str, use_chat_template: bool) -> int:
    formatted = format_prompt(tokenizer, prompt_text, use_chat_template)
    return int(len(tokenizer(formatted)["input_ids"]))


def build_prompt_for_target(tokenizer, target_len: int, use_chat_template: bool) -> str:
    base = " benchmark"
    lo = 1
    hi = 2
    while prompt_len(tokenizer, base * hi, use_chat_template) < target_len:
        lo = hi
        hi *= 2
    best = base * hi
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = base * mid
        cand_len = prompt_len(tokenizer, candidate, use_chat_template)
        if cand_len >= target_len:
            best = candidate
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def extract_json_block(stdout_text: str) -> Dict[str, object]:
    match = re.search(r"\{[\s\S]*\}", stdout_text)
    if not match:
        raise ValueError("Could not locate JSON profile payload in script output.")
    return json.loads(match.group(0))


def write_subprocess_error_log(
    error_dir: str,
    run_name: str,
    cmd: List[str],
    stdout_text: str,
    stderr_text: str,
    returncode: Optional[int],
) -> str:
    os.makedirs(error_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(error_dir, f"{run_name}_{stamp}.log")
    lines = [
        f"run_name: {run_name}",
        f"returncode: {returncode}",
        f"command: {shlex.join(cmd)}",
        "",
        "=== STDOUT ===",
        stdout_text or "<empty>",
        "",
        "=== STDERR ===",
        stderr_text or "<empty>",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def run_profile(
    config: RunConfig,
    run_name: str,
    prompt_text: str,
    gen_length: int,
    batch_size: int,
    warmup: int,
    repetitions: int,
    use_profiler: bool,
    profiler_dir: str,
    error_dir: str,
    do_compile: bool,
    block_size: int,
    threshold: float,
) -> Dict[str, object]:
    cmd = [
        sys.executable,
        config.script_path,
        "--model",
        config.model_path,
        "--prompt-mode",
        "prompted",
        "--prompt",
        prompt_text,
        "--gen-length",
        str(gen_length),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(warmup),
        "--repetitions",
        str(repetitions),
    ]
    if config.use_chat_template:
        cmd.append("--use-chat-template")
    if do_compile:
        cmd.append("--do-compile")
    if use_profiler:
        cmd.extend(["--torch-profiler", "--profiler-dir", profiler_dir])
    if config.offload_dir:
        cmd.extend(["--offload-dir", config.offload_dir])
    if config.model_key == "fast_dllm":
        cmd.extend(["--block-size", str(block_size), "--threshold", str(threshold)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        log_path = write_subprocess_error_log(
            error_dir=error_dir,
            run_name=run_name,
            cmd=cmd,
            stdout_text=exc.stdout or "",
            stderr_text=exc.stderr or "",
            returncode=exc.returncode,
        )
        raise RuntimeError(
            f"{run_name} failed with exit code {exc.returncode}. "
            f"Subprocess output was written to {log_path}."
        ) from exc

    payload = extract_json_block(proc.stdout)
    trace_match = re.search(r"TensorBoard trace dir(?: \([^)]+\))?:\s*(.+)", proc.stdout)
    if trace_match:
        payload["trace_dir"] = trace_match.group(1).strip()
    payload["raw_stdout"] = proc.stdout
    return payload


def write_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-model context-length benchmark sweeps.")
    parser.add_argument("--context-bins", default="128,256,512,1024,2048,4096")
    parser.add_argument("--profile-bins", default="minmidmax", help="none|all|minmidmax|comma-separated bins")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--do-compile", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--results-dir", default="benchmarks/results")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--fast-dllm-model", default="Efficient-Large-Model/Fast_dLLM_v2_1.5B")
    parser.add_argument("--qwen-offload-dir", default=None)
    parser.add_argument("--fast-dllm-offload-dir", default=None)
    parser.add_argument("--fast-dllm-block-size", type=int, default=32)
    parser.add_argument("--fast-dllm-threshold", type=float, default=0.9)
    args = parser.parse_args()

    context_bins = parse_bins(args.context_bins)
    profile_bins = parse_profile_bins(args.profile_bins, context_bins)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.results_dir, stamp)
    traces_root = os.path.join(run_root, "traces")
    error_root = os.path.join(run_root, "errors")
    os.makedirs(traces_root, exist_ok=True)

    configs = [
        RunConfig(
            model_key="qwen",
            script_path="Qwen/profile_inference.py",
            model_path=args.qwen_model,
            use_chat_template=args.use_chat_template,
            offload_dir=args.qwen_offload_dir,
        ),
        RunConfig(
            model_key="fast_dllm",
            script_path="fast_dllm/profile_inference.py",
            model_path=args.fast_dllm_model,
            use_chat_template=args.use_chat_template,
            offload_dir=args.fast_dllm_offload_dir,
        ),
    ]

    rows: List[Dict[str, object]] = []
    prompt_cache: Dict[str, Dict[int, str]] = {}

    for config in configs:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        prompt_cache[config.model_key] = {}
        for target_context in context_bins:
            prompt_text = build_prompt_for_target(tokenizer, target_context, config.use_chat_template)
            prompt_cache[config.model_key][target_context] = prompt_text

    for config in configs:
        for target_context in context_bins:
            prompt_text = prompt_cache[config.model_key][target_context]
            baseline = run_profile(
                config=config,
                run_name=f"{config.model_key}_baseline_ctx{target_context}",
                prompt_text=prompt_text,
                gen_length=args.gen_length,
                batch_size=args.batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
                use_profiler=False,
                profiler_dir=traces_root,
                error_dir=error_root,
                do_compile=args.do_compile,
                block_size=args.fast_dllm_block_size,
                threshold=args.fast_dllm_threshold,
            )
            baseline["target_context_len"] = target_context
            baseline["run_type"] = "baseline"
            baseline["model_key"] = config.model_key
            rows.append(baseline)

            if target_context in profile_bins:
                traced = run_profile(
                    config=config,
                    run_name=f"{config.model_key}_profiled_ctx{target_context}",
                    prompt_text=prompt_text,
                    gen_length=args.gen_length,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    use_profiler=True,
                    profiler_dir=traces_root,
                    error_dir=error_root,
                    do_compile=args.do_compile,
                    block_size=args.fast_dllm_block_size,
                    threshold=args.fast_dllm_threshold,
                )
                traced["target_context_len"] = target_context
                traced["run_type"] = "profiled"
                traced["model_key"] = config.model_key
                rows.append(traced)

    manifest = {
        "run_root": run_root,
        "created_at": stamp,
        "args": vars(args),
        "context_bins": context_bins,
        "profile_bins": sorted(profile_bins),
        "rows": rows,
    }

    write_json(os.path.join(run_root, "summary.json"), manifest)
    write_csv(os.path.join(run_root, "summary.csv"), rows)
    print(json.dumps({"run_root": run_root, "row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
