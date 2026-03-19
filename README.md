# dlm-research

## Benchmark Workflow

In the `fast_dllm` and `Qwen` directories, there are `demo.py` and `profile_inference.py`. Run `demo.py` first so HuggingFace downloads and caches the model:

```bash
python Qwen/demo.py
python fast_dllm/demo.py
```

Then run the profiling scripts directly, for example:

```bash
python fast_dllm/profile_inference.py
```

These scripts create `tb_profiles/`, which contains trace and profiler outputs that can be loaded in TensorBoard or Perfetto.

Notable arguments:
- `--warmup`
- `--repetitions`
- `--torch-profiler`
- `--do-compile`

Example:

```bash
python fast_dllm/profile_inference.py --torch-profiler --do-compile --warmup 3 --repetitions 1 --gen-length 128 --prompt-mode prompted --prompt "What is the capital of France?"
```

`benchmarks/context_sweep.py` builds prompts at different target context lengths and calls the per-model `profile_inference.py` scripts as subprocesses with those prompts. The output is saved in `benchmarks/results/`

Example:

```bash
python benchmarks/context_sweep.py \
  --context-bins 128,256,512,1024,2048,4096 \
  --profile-bins all \
  --gen-length 128 \
  --batch-size 1 \
  --warmup 3 \
  --repetitions 1
```

`--profile-bins` controls which context lengths also get a Torch profiler run. If it is not set, the script runs the requested context bins only. If it is set, the script still runs the requested context bins, then does an additional pass on the selected profile bins with `--torch-profiler`.

After running the baseline, run the same sweep again with `torch.compile`:

```bash
python benchmarks/context_sweep.py \
  --context-bins 128,256,512,1024,2048,4096 \
  --profile-bins all \
  --gen-length 128 \
  --batch-size 1 \
  --warmup 3 \
  --repetitions 1 \
  --do-compile
```

See `benchmarks/README.md` for more commands and outputs.
