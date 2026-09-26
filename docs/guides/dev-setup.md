# Development setup

## PC with NVIDIA GPU (primary ML machine): use WSL2

Why WSL2: CI, Docker and cloud GPUs are all Linux. Developing in the same OS removes a whole class of "works on my machine" problems, and PyTorch's default Linux wheels include CUDA.

1. **Windows side**
   - Install the latest NVIDIA Game Ready or Studio driver. Do *not* install a CUDA toolkit inside WSL; the Windows driver exposes the GPU to WSL.
   - Install WSL2 with Ubuntu 24.04: `wsl --install -d Ubuntu-24.04`
   - Install VS Code with the *WSL* extension, then open the repo with `code .` from inside WSL.
   - Optional: Docker Desktop with the WSL2 backend (for `docker/`).
2. **Inside WSL (Ubuntu)**
   ```bash
   sudo apt update && sudo apt install -y git make build-essential
   curl -LsSf https://astral.sh/uv/install.sh | sh        # uv (Python + packages)
   git clone <repo-url> && cd <repo>
   make setup                                             # core + dev + api extras, pre-commit hooks
   uv sync --extra api --extra neuro --extra dl --extra tracking   # full ML stack when you need it
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   make check && make smoke
   ```
3. **Keep datasets on the Linux filesystem** (`~/…`), not under `/mnt/c/…`. Cross-filesystem IO is roughly 10× slower.
4. **MLflow UI** (after WP-0.7): `uv run mlflow ui --backend-store-uri ./mlruns`, then open http://localhost:5000 from Windows.

## Mac (frontend, API, product testing)

```bash
brew install uv node@22 git make
make setup
make api        # FastAPI on :8000
make web        # Next.js on :3000
```

On Apple Silicon, PyTorch uses the `mps` device for small experiments. Heavy training belongs on the PC.

## Cloud GPUs (only when the local GPU is not enough)

- Use the same Docker image (`docker/ml.Dockerfile`; GPU variant in WP-3.3) and the same experiment configs. Results are only comparable if the environment is identical.
- Keep datasets in a private bucket (C1/C3 rules), and shut down instances after use.
- Record the instance type in the run manifest (`platform` field).

## Editor

VS Code extensions:

- Python
- Pylance
- Ruff
- Jupyter
- ESLint
- Prettier (optional)
- GitLens

`.editorconfig` sets whitespace conventions.
