import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLaDA GGUF with llama.cpp diffusion CLI (Q8_0 by default)."
    )
    parser.add_argument(
        "--repo-id",
        default="mradermacher/LLaDA-8B-Base-GGUF",
        help="Hugging Face model repo containing GGUF files.",
    )
    parser.add_argument(
        "--filename",
        default="LLaDA-8B-Base.Q8_0.gguf",
        help="GGUF file name inside the repo.",
    )
    parser.add_argument(
        "--prompt",
        default="Give me a short introduction to diffusion language models.",
        help="Prompt text.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=0)

    # LLaDA diffusion-specific knobs
    parser.add_argument("--diffusion-steps", type=int, default=128)
    parser.add_argument(
        "--diffusion-block-length",
        type=int,
        default=32,
        help="LLaDA block length for generation.",
    )
    parser.add_argument(
        "--diffusion-algorithm",
        type=int,
        default=4,
        help="0=ORIGIN,1=ENTROPY,2=MARGIN,3=RANDOM,4=CONFIDENCE/LOW_CONFIDENCE",
    )
    parser.add_argument("--diffusion-visual", action="store_true")

    parser.add_argument(
        "--llama-diffusion-cli",
        default=None,
        help="Path to llama-diffusion-cli binary. If omitted, auto-detect from PATH/common build paths.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="HF token (defaults to HF_TOKEN env var).",
    )
    return parser


def download_gguf(repo_id: str, filename: str, token: Optional[str]) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, token=token)


def find_diffusion_cli(explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        p = Path(explicit_path)
        return str(p) if p.is_file() else None

    in_path = shutil.which("llama-diffusion-cli")
    if in_path:
        return in_path

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "llama.cpp/build/bin/llama-diffusion-cli",
        Path.cwd() / "llama.cpp/build/bin/llama-diffusion-cli",
        Path.cwd() / "build/bin/llama-diffusion-cli",
        Path("/usr/local/bin/llama-diffusion-cli"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def main() -> None:
    args = build_parser().parse_args()

    # llama-diffusion-cli uses internal max_length=512 by default; for block mode,
    # steps must be divisible by num_blocks (= max_length / block_length).
    if args.diffusion_block_length > 0 and 512 % args.diffusion_block_length == 0:
        num_blocks = 512 // args.diffusion_block_length
        if args.diffusion_steps % num_blocks != 0:
            raise SystemExit(
                f"Invalid combination: --diffusion-steps {args.diffusion_steps} "
                f"must be divisible by {num_blocks} when --diffusion-block-length "
                f"is {args.diffusion_block_length}."
            )

    model_path = download_gguf(args.repo_id, args.filename, args.hf_token)
    cli_path = find_diffusion_cli(args.llama_diffusion_cli)

    if not cli_path:
        raise SystemExit(
            "Could not find llama-diffusion-cli.\n"
            "Build/install latest llama.cpp with diffusion support, then run again.\n"
            "Example usage:\n"
            "  llama-diffusion-cli -m <model.gguf> -p \"...\" --diffusion-steps 128 --diffusion-block-length 32"
        )

    cmd = [
        cli_path,
        "-m",
        model_path,
        "-p",
        args.prompt,
        "-n",
        str(args.max_new_tokens),
        "-c",
        str(args.n_ctx),
        "-b",
        str(args.n_batch),
        "-ub",
        str(args.n_ubatch),
        "-ngl",
        str(args.n_gpu_layers),
        "--temp",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--diffusion-algorithm",
        str(args.diffusion_algorithm),
    ]

    cmd.extend(["--diffusion-block-length", str(args.diffusion_block_length)])

    if args.diffusion_visual:
        cmd.append("--diffusion-visual")

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
