import argparse
import json
import os
from pathlib import Path
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

@dataclass
class ProfileResult:
    prompt_len: int
    gen_length: int
    batch_size: int
    total_time_s: float
    per_token_time_ms: float
    tokens_per_s: float
    max_mem_alloc_mb: float
    max_mem_reserved_mb: float


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    repetitions: int,
    base_dir: str,
    do_compile: bool,
) -> str:
    model_name = _sanitize_model_name(model_path)
    args = f"{model_name}_{prompt_mode}_gl{gen_length}_rep{repetitions}"
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args}/plugins/profile/{time_stamp}"
    if do_compile:
        run_name += "_compiled"
    return os.path.join(base_dir, run_name)

def _format_prompt(tokenizer, prompt_text: str, use_chat_template: bool) -> str:
    if use_chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    if not prompt_text.strip():
        if tokenizer.eos_token:
            return tokenizer.eos_token
        return " "
    return prompt_text

def _load_model(model_path: str, do_compile: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    if do_compile: model.compile()
    return model


def run_profile(
    model_path: str,
    prompt_mode: str,
    prompt_text: str,
    use_chat_template: bool,
    gen_length: int,
    block_size: int,
    threshold: float,
    batch_size: int,
    warmup: int,
    repetitions: int,
    use_torch_profiler: bool,
    profiler_dir: Optional[str],
    do_compile: bool,
    offload_dir: Optional[str],
) -> Tuple[ProfileResult, Optional[str]]:
    model = _load_model(model_path=model_path, do_compile=do_compile)
    tokenizer = None
    can_skip_tokenizer = (not use_chat_template) and (not prompt_text.strip())

    if can_skip_tokenizer:
        token_id = (
            model.config.bos_token_id
            if model.config.bos_token_id is not None
            else model.config.eos_token_id
        )
        if token_id is None:
            token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        input_ids = torch.full((batch_size, 1), token_id, dtype=torch.long, device=model.device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, device=model.device),
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = _format_prompt(tokenizer, prompt_text, use_chat_template)
        model_inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_time = 0.0
    run_dir = None

    if use_torch_profiler:
        base_prof_dir = profiler_dir or "./tb_profiles"
        run_dir = _build_run_dir(
            model_path=model_path,
            prompt_mode=prompt_mode,
            gen_length=gen_length,
            repetitions=repetitions,
            base_dir=base_prof_dir,
            do_compile=do_compile,
        )
        os.makedirs(run_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        wait_steps = 1
        schedule = torch.profiler.schedule(wait=wait_steps, warmup=warmup, active=repetitions, repeat=1)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(run_dir, worker_name=None)
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=False,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) as prof:
            active_start_step = wait_steps + warmup
            total_profile_steps = active_start_step + repetitions
            for step_idx in range(total_profile_steps):
                _sync()
                start = time.perf_counter()
                model.generate(**model_inputs, max_new_tokens=gen_length, small_block_size=block_size, threshold=threshold)
                _sync()
                end = time.perf_counter()
                if step_idx >= active_start_step:
                    total_time += (end - start)
                prof.step()
    else:
        for _ in range(repetitions):
            _sync()
            start = time.perf_counter()
            model.generate(**model_inputs, max_new_tokens=gen_length, small_block_size=block_size, threshold=threshold)
            _sync()
            end = time.perf_counter()
            total_time += (end - start)

    avg_time_s = total_time / max(repetitions, 1)
    total_gen_tokens = gen_length * batch_size
    per_token_time_ms = (avg_time_s * 1000.0 / gen_length) if gen_length > 0 else 0.0
    tokens_per_s = (total_gen_tokens / avg_time_s) if avg_time_s > 0 else 0.0

    max_alloc = 0.0
    max_reserved = 0.0
    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    result = ProfileResult(
        prompt_len=int(model_inputs["input_ids"].shape[1]),
        gen_length=gen_length,
        batch_size=batch_size,
        total_time_s=avg_time_s,
        per_token_time_ms=per_token_time_ms,
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
    block_size: int,
    threshold: float,
    batch_size: int,
    warmup: int,
    repetitions: int,
    use_torch_profiler: bool,
    profiler_dir: Optional[str],
    do_compile: bool,
    offload_dir: Optional[str],
) -> Tuple[ProfileResult, ProfileResult, Optional[str], Optional[str]]:
    prompted_result, prompted_trace = run_profile(
        model_path=model_path,
        prompt_mode="prompted",
        prompt_text=prompted_text,
        use_chat_template=use_chat_template,
        gen_length=gen_length,
        block_size=block_size,
        threshold=threshold,
        batch_size=batch_size,
        warmup=warmup,
        repetitions=repetitions,
        use_torch_profiler=use_torch_profiler,
        profiler_dir=profiler_dir,
        do_compile=do_compile,
        offload_dir=offload_dir,
    )
    prompt_free_result, prompt_free_trace = run_profile(
        model_path=model_path,
        prompt_mode="free",
        prompt_text=prompt_free_text,
        use_chat_template=False,
        gen_length=gen_length,
        block_size=block_size,
        threshold=threshold,
        batch_size=batch_size,
        warmup=warmup,
        repetitions=repetitions,
        use_torch_profiler=use_torch_profiler,
        profiler_dir=profiler_dir,
        do_compile=do_compile,
        offload_dir=offload_dir,
    )
    return prompted_result, prompt_free_result, prompted_trace, prompt_free_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Fast_dLLM inference (TensorBoard torch.profiler)"
    )
    parser.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_1.5B", help="HF path or local path")
    parser.add_argument("--prompt-mode", choices=["free", "prompted", "dual"], default="free")
    parser.add_argument("--prompt", default="Hello", help="Prompt text for prompted mode")
    parser.add_argument("--prompt-free-text", default="", help="Prompt text for prompt-free mode")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--do-compile", action="store_true")
    parser.add_argument("--profiler-dir", default=None)
    parser.add_argument("--offload-dir", default=None, help="Offload folder for device_map=auto")
    args = parser.parse_args()

    if args.prompt_mode == "dual":
        prompted_result, prompt_free_result, prompted_trace, prompt_free_trace = run_dual_mode(
            model_path=args.model,
            prompted_text=args.prompt,
            prompt_free_text=args.prompt_free_text,
            use_chat_template=args.use_chat_template,
            gen_length=args.gen_length,
            block_size=args.block_size,
            threshold=args.threshold,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
            use_torch_profiler=args.torch_profiler,
            profiler_dir=args.profiler_dir,
            do_compile=args.do_compile,
            offload_dir=args.offload_dir,
        )
    else:
        result, trace_path = run_profile(
            model_path=args.model,
            prompt_mode=args.prompt_mode,
            prompt_text=args.prompt if args.prompt_mode == "prompted" else args.prompt_free_text,
            use_chat_template=args.use_chat_template if args.prompt_mode == "prompted" else False,
            gen_length=args.gen_length,
            block_size=args.block_size,
            threshold=args.threshold,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
            use_torch_profiler=args.torch_profiler,
            profiler_dir=args.profiler_dir,
            do_compile=args.do_compile,
            offload_dir=args.offload_dir,
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
