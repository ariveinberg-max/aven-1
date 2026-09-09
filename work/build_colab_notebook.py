"""Builds Aven-1-Colab.ipynb by embedding the current source files verbatim.
Run this whenever brain.py/tokenizer.py/train.py/tracking.py change,
to keep the notebook in sync with the real project.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def code(source_lines):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source_lines}


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [text]}


def writefile_cell(filename):
    text = (ROOT / filename).read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    return code([f'%%writefile {filename}\n'] + lines)


cells = [
    md('# Aven-1 — Cloud GPU Training (Colab)\n\n'
       'Free-tier Colab GPU (T4) training for the from-scratch Aven-1 model. '
       'This notebook embeds the exact same `brain.py` / `tokenizer.py` / `train.py` / `tracking.py` '
       'you run locally — nothing different, just running on a faster GPU for free.\n\n'
       '**One-time setup per session:**\n'
       '1. Runtime -> Change runtime type -> GPU (T4 is fine, free tier)\n'
       '2. Run all cells top to bottom\n'
       '3. When prompted, upload your `checkpoints/` folder (from your Mac) into the Drive folder — '
       'or skip that step to train a brand new model from scratch here.\n'),

    md('## 1. Confirm GPU is attached'),
    code(['!nvidia-smi\n']),

    md('## 2. Mount Google Drive (persists your checkpoint across sessions)'),
    code(["from google.colab import drive\n",
          "drive.mount('/content/drive')\n",
          "import os\n",
          "WORKDIR = '/content/drive/MyDrive/aven-1'\n",
          "os.makedirs(WORKDIR, exist_ok=True)\n",
          "os.makedirs(f'{WORKDIR}/data', exist_ok=True)\n",
          "os.makedirs(f'{WORKDIR}/checkpoints', exist_ok=True)\n",
          "%cd {WORKDIR}\n"]),

    md('## 3. Write the project source files (kept identical to your local copy)'),
    writefile_cell('brain.py'),
    writefile_cell('tokenizer.py'),
    writefile_cell('train.py'),
    writefile_cell('tracking.py'),

    md('## 4. Install dependencies (torch/numpy already present on Colab)'),
    code(['!pip install -q wandb\n']),

    md('## 5. Log in to Weights & Biases\n\n'
       'Paste your API key when prompted (from https://wandb.ai/authorize). '
       'This is the same free account you already use locally — runs from here land in the same `aven-1` project.'),
    code(['import wandb\n', 'wandb.login()\n']),

    md('## 6. Bring your data and checkpoint over\n\n'
       'One-time only (Drive persists after this): open your Google Drive in a browser, '
       'go to the `aven-1` folder this notebook just created, and **drag in**:\n'
       '- your local `data/instructions.txt` (or `data/training.txt`) into `aven-1/data/`\n'
       '- your local `checkpoints/latest.pt`, `checkpoints/tokenizer.json`, `checkpoints/status.json` '
       'into `aven-1/checkpoints/` — **only if you want to continue your existing model**. '
       'Skip this to train a brand new model from scratch on this GPU instead.\n\n'
       'Run the next cell after uploading to confirm the files are there.'),
    code(["!ls -la {WORKDIR}/data {WORKDIR}/checkpoints\n"]),

    md('## 7. Train\n\n'
       'This is the exact same `train.py` you run locally — same flags, same behavior, just `--device cuda`. '
       'Adjust `--data`, `--steps`, `--finetune` as needed. Examples:\n\n'
       '```\n'
       '# Continue fine-tuning your existing model on more instruction data:\n'
       '!python train.py --data data/instructions.txt --finetune --steps 3000 --device cuda --wandb\n\n'
       '# Fresh pretraining run from scratch:\n'
       '!python train.py --data data/training.txt --steps 3000 --device cuda --width 512 --layers 6 --heads 8 --wandb\n'
       '```'),
    code(['!python train.py --data data/instructions.txt --finetune --steps 3000 --device cuda --wandb\n']),

    md('## 8. Get your checkpoint back to your Mac\n\n'
       'It already lives in Google Drive (`aven-1/checkpoints/`) since we mounted Drive — '
       'just download `latest.pt`, `tokenizer.json`, and `status.json` from the Drive web UI '
       'and drop them into your local `checkpoints/` folder to keep using it with the dashboard and chat page.'),
]

nb = {
    'cells': cells,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'name': 'Aven-1-Colab.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 0,
}

out = ROOT / 'Aven-1-Colab.ipynb'
out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Wrote {out}')
