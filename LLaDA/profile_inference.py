import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from generate import generate


@dataclass
class ProfileResult:
    prompt_len: int
    gen_length: int
    steps: int
    block_length: int
    batch_size: int
    total_time_s: float
    per_step_time_ms: float
    tokens_per_s: float
    max_mem_alloc_mb: float
    max_mem_reserved_mb: float


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _format_prompt(tokenizer, prompt_text: str, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    if not prompt_text.strip():
        if tokenizer.eos_token:
            prompt_text = tokenizer.eos_token
        else:
            prompt_text = " "
    input_ids = tokenizer(prompt_text)["input_ids"]
    return torch.tensor(input_ids).unsqueeze(0)


def _sanitize_model_name(model_path: str) -> str:
    base = model_path.rstrip("/").split("/")[-1]
    cleaned = []
    for ch in base:
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned) or "model"


def _build_run_dir(
    model_path: str,
    prompt_mode: str,
    gen_length: int,
    steps: int,
    block_length: int,
    repetitions: int,
    base_dir: str,
    do_compile: bool,
) -> str:
    model_name = _sanitize_model_name(model_path)
    args = (
        f"{model_name}_{prompt_mode}_gl{gen_length}_s{steps}_"
        f"bl{block_length}_rep{repetitions}"
    )
    tb_req = "plugins/profile"
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f'{args}/{tb_req}/{time_stamp}'
    run_name += '_compiled' if do_compile else ''
    return os.path.join(base_dir, run_name)

def run_profile(
    model_path: str,
    prompt_mode: str,
    prompt_text: str,
    use_chat_template: bool,
    gen_length: int,
    steps: int,
    block_length: int,
    batch_size: int,
    device: str,
    warmup: int,
    repetitions: int,
    use_torch_profiler: bool,
    profiler_dir: Optional[str],
    do_compile: bool,
) -> Tuple[ProfileResult, Optional[str]]:
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    if do_compile: model.compile()

    # Build prompt batch
    prompt_ids = _format_prompt(tokenizer, prompt_text, use_chat_template)
    prompt_ids = prompt_ids.to(device)
    if batch_size > 1:
        prompt_ids = prompt_ids.repeat(batch_size, 1)

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Warmup
    for _ in range(warmup):
        _sync()
        _ = generate(
            model,
            prompt_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
        _sync()

    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Timed runs
    total_time = 0.0
    run_dir = None

    if use_torch_profiler:
        base_prof_dir = profiler_dir or "./tb_profiles"
        run_dir = _build_run_dir(
            model_path=model_path,
            prompt_mode=prompt_mode,
            gen_length=gen_length,
            steps=steps,
            block_length=block_length,
            repetitions=repetitions,
            base_dir=base_prof_dir,
            do_compile=do_compile,
        )
        os.makedirs(run_dir, exist_ok=True)
        activities = (
            [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            if device.type == "cuda"
            else [torch.profiler.ProfilerActivity.CPU]
        )
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=repetitions, repeat=1)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(run_dir, worker_name=None)
        with torch.profiler.profile(activities=activities,
                                    schedule=schedule,
                                    on_trace_ready=on_trace_ready,
                                    record_shapes=False,
                                    profile_memory=True,
                                    with_stack=True,
                                    with_flops=True,
                                    with_modules=True,
                                    ) as prof:
            for _ in range(repetitions):
                _sync()
                start = time.perf_counter()
                _ = generate(
                    model,
                    prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                )
                _sync()
                end = time.perf_counter()
                total_time += (end - start)
                prof.step()
    else:
        for _ in range(repetitions):
            _sync()
            start = time.perf_counter()
            _ = generate(
                model,
                prompt_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )
            _sync()
            end = time.perf_counter()
            total_time += (end - start)

    # Per-step time: steps are per block; total denoise steps = steps
    # In generate(), steps is divided by num_blocks. Total iterations = steps.
    per_step_time_ms = (total_time / repetitions) * 1000.0 / float(steps)

    # Throughput: generated tokens / second
    total_gen_tokens = gen_length * batch_size
    tokens_per_s = total_gen_tokens / (total_time / repetitions)

    max_alloc = 0.0
    max_reserved = 0.0
    if device.type == "cuda":
        max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    result = ProfileResult(
        prompt_len=int(prompt_ids.shape[1]),
        gen_length=gen_length,
        steps=steps,
        block_length=block_length,
        batch_size=batch_size,
        total_time_s=total_time / repetitions,
        per_step_time_ms=per_step_time_ms,
        tokens_per_s=tokens_per_s,
        max_mem_alloc_mb=max_alloc,
        max_mem_reserved_mb=max_reserved,
    )
    return result, run_dir


def run_dual_mode(
    model_path: str,
    prompted_text: str,
    prompt_free_text: str,
    use_chat_template: bool,
    gen_length: int,
    steps: int,
    block_length: int,
    batch_size: int,
    device: str,
    warmup: int,
    repetitions: int,
    use_torch_profiler: bool,
    profiler_dir: Optional[str],
    do_compile: bool,
) -> Tuple[ProfileResult, ProfileResult, Optional[str], Optional[str]]:
    prompted_result, prompted_trace = run_profile(
        model_path=model_path,
        prompt_mode="prompted",
        prompt_text=prompted_text,
        use_chat_template=use_chat_template,
        gen_length=gen_length,
        steps=steps,
        block_length=block_length,
        batch_size=batch_size,
        device=device,
        warmup=warmup,
        repetitions=repetitions,
        use_torch_profiler=use_torch_profiler,
        profiler_dir=profiler_dir,
        do_compile=do_compile,
    )
    prompt_free_result, prompt_free_trace = run_profile(
        model_path=model_path,
        prompt_mode="free",
        prompt_text=prompt_free_text,
        use_chat_template=False,
        gen_length=gen_length,
        steps=steps,
        block_length=block_length,
        batch_size=batch_size,
        device=device,
        warmup=warmup,
        repetitions=repetitions,
        use_torch_profiler=use_torch_profiler,
        profiler_dir=profiler_dir,
        do_compile=do_compile,
    )
    return prompted_result, prompt_free_result, prompted_trace, prompt_free_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile LLaDA inference only (TensorBoard torch.profiler)"
    )
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct", help="HF path or local path")
    parser.add_argument("--prompt-mode", choices=["free", "prompted", "dual"], default="free")
    parser.add_argument("--prompt", default="Hello", help="Prompt text for prompted mode")
    parser.add_argument("--prompt-free-text", default="", help="Prompt text for prompt-free mode")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--do-compile", action="store_true")
    parser.add_argument("--profiler-dir", default=None)
    args = parser.parse_args()

    if args.prompt_mode == "dual":
        prompted_result, prompt_free_result, prompted_trace, prompt_free_trace = run_dual_mode(
            model_path=args.model,
            prompted_text=args.prompt,
            prompt_free_text=args.prompt_free_text,
            use_chat_template=args.use_chat_template,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            batch_size=args.batch_size,
            device=args.device,
            warmup=args.warmup,
            repetitions=args.repetitions,
            use_torch_profiler=args.torch_profiler,
            profiler_dir=args.profiler_dir,
            do_compile=args.do_compile
        )
    else:
        result, trace_path = run_profile(
            model_path=args.model,
            prompt_mode=args.prompt_mode,
            prompt_text=args.prompt if args.prompt_mode == "prompted" else args.prompt_free_text,
            use_chat_template=args.use_chat_template if args.prompt_mode == "prompted" else False,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            batch_size=args.batch_size,
            device=args.device,
            warmup=args.warmup,
            repetitions=args.repetitions,
            use_torch_profiler=args.torch_profiler,
            profiler_dir=args.profiler_dir,
            do_compile=args.do_compile
        )

    if args.prompt_mode == "dual":
        print("Profile result: prompted")
        print(json.dumps(prompted_result.__dict__, indent=2))
        print("Profile result: prompt-free")
        print(json.dumps(prompt_free_result.__dict__, indent=2))
        if prompted_trace:
            print(f"TensorBoard trace dir (prompted): {prompted_trace}")
        if prompt_free_trace:
            print(f"TensorBoard trace dir (prompt-free): {prompt_free_trace}")
    else:
        print("Profile result:")
        print(json.dumps(result.__dict__, indent=2))
        if trace_path:
            print(f"TensorBoard trace dir: {trace_path}")


if __name__ == "__main__":
    main()
