# Setup the `dl` environment on a new server

This repo ships a fully pinned conda/pip lock so you can reproduce the
training environment (Python 3.10 + PyTorch 2.9.1+cu128 + LLaMA-Factory
editable) in two commands.

## 1. Prerequisites on the new machine

- NVIDIA driver recent enough for CUDA 12.8 runtime wheels (driver >= 525.60
  works, newer is safer).
- Anaconda / Miniconda installed.
- This repo cloned somewhere, e.g. `/path/to/Debug_LM`.

Check driver: `nvidia-smi` (top-right CUDA version should be >= 12.0).

## 2. Create the env

```bash
cd /path/to/Debug_LM
conda env create -f environment.yml
conda activate dl
pip install -e .        # installs llamafactory from this repo (editable)
```

That's it. `llamafactory-cli` will be available on PATH, and `python -c
"import torch; print(torch.cuda.is_available())"` should print `True`.

## 3. Update later

If you add/remove packages, re-freeze on the source machine:

```bash
conda activate dl
pip freeze | grep -Ev '^(-e |llamafactory @|llama-factory @)' > requirements.lock.txt
```

## 4. Troubleshooting

- **`torch.cuda.is_available() == False`**: the PyPI torch wheel fell back
  to a variant that doesn't match your system. Open `requirements.lock.txt`
  and uncomment the `--extra-index-url https://download.pytorch.org/whl/cu128`
  line, then reinstall torch:
  ```bash
  pip install --force-reinstall --no-deps torch==2.9.1 \
      --extra-index-url https://download.pytorch.org/whl/cu128
  ```
- **Different CUDA (e.g. only CUDA 12.1 driver)**: pin to a matching torch
  build. For cu121 use `torch==2.5.1` with
  `--extra-index-url https://download.pytorch.org/whl/cu121`, and be aware
  that `deepspeed==0.14.4` should still work but some `nvidia-*-cu12`
  wheel versions will need to match.
- **`deepspeed` JIT build errors**: set `export DS_BUILD_OPS=0` to force
  pure-Python mode, or make sure `gcc`, `g++`, and `ninja` are available on
  the machine.
- **Rebuild from scratch**: `conda env remove -n dl && conda env create -f environment.yml`.

## 5. What is NOT pinned here

The following are intentionally NOT in the lockfile and may need separate
installs if you need them:

- `flash-attn` — not installed in the source env. If you want it:
  `pip install flash-attn --no-build-isolation` (needs matching CUDA toolchain).
- `bitsandbytes` / `vllm` / `xformers` — not installed in the source env.
- System-level CUDA toolkit / `nvcc` — only required if you JIT-compile
  custom kernels (e.g. flash-attn). Regular training + DeepSpeed ZeRO-3
  work fine with just the driver + pip-installed `nvidia-*-cu12` wheels.
