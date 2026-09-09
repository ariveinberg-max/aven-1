"""Builds Aven-1-Kaggle.ipynb by embedding the current source files verbatim.
Run whenever brain.py/tokenizer.py/train.py/tracking.py change, same as build_colab_notebook.py.
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
    md('# Aven-1 — Cloud GPU Training (Kaggle)\n\n'
       'A second free-GPU option alongside the Colab notebook — Kaggle gives ~30 hrs/week of '
       'guaranteed free GPU quota (T4/P100), more predictable than Colab\'s free tier session limits. '
       'Same exact source code as your local project, just running on Kaggle\'s GPU.\n\n'
       '**One-time Kaggle setup, before this notebook will work:**\n'
       '1. Kaggle requires phone number verification to enable GPU/internet on notebooks — '
       'Settings (your avatar) -> Settings -> Phone Verification, if you haven\'t already.\n'
       '2. In *this* notebook: the panel on the right -> **Accelerator** -> pick **GPU T4 x2** (or whatever GPU is offered).\n'
       '3. Same panel -> **Internet** -> turn **On** (needed for `pip install wandb` and `wandb login`).\n\n'
       '**Persistence works differently from Colab (no Google Drive here):** Kaggle notebooks use '
       '"Datasets" as the equivalent of a Drive folder. You upload your `data/` and `checkpoints/` '
       'folders once as a Kaggle Dataset, attach it as input to this notebook, and when you want to '
       'save results forward, you create a new dataset version from this notebook\'s output. '
       'Slower than Colab\'s automatic Drive sync, but works fine — instructions are in the cells below.'),

    md('## 1. Confirm GPU is attached\n(If this errors, go set the Accelerator to GPU in the panel on the right first.)'),
    code(['!nvidia-smi\n']),

    md('## 2. Write the project source files (kept identical to your local copy)'),
    writefile_cell('brain.py'),
    writefile_cell('tokenizer.py'),
    writefile_cell('train.py'),
    writefile_cell('tracking.py'),

    md('## 3. Install dependencies'),
    code(['!pip install -q wandb\n']),

    md('## 4. Log in to Weights & Biases\n\n'
       'Same free account as your local setup and the Colab notebook — paste your API key from '
       'https://wandb.ai/authorize when prompted. Runs from here land in the same `aven-1` project.'),
    code(['import wandb\n', 'wandb.login()\n']),

    md('## 5. Bring in your data and checkpoint\n\n'
       '**One-time setup (do this once, reuse the dataset every session after):**\n'
       '1. On your Mac, zip your local `data/` folder (at least `data/training.txt` or `data/instructions.txt`) '
       'and your `checkpoints/` folder (`latest.pt`, `tokenizer.json`, `status.json`) — or just the whole project folder.\n'
       '2. Go to kaggle.com/datasets -> **New Dataset** -> upload those files/folders -> give it a name '
       'like `aven-1-data`.\n'
       '3. Back in this notebook: click **Add Input** (right panel) -> search for the dataset you just made '
       '-> add it. It will appear read-only under `/kaggle/input/<your-dataset-name>/`.\n\n'
       'The cell below defaults to your flat-file dataset at `/kaggle/input/datasets/ariveinbergs/aven-1-data/`. '
       'It also accepts nested `data/` and `checkpoints/` folders and preserves existing working files on rerun.'),
    code(['from pathlib import Path\n', 'import shutil\n', '\n', "DATASET_PATH = Path('/kaggle/input/datasets/ariveinbergs/aven-1-data')\n", '# Accept the current flat upload and older data/checkpoints folder layouts.\n', "required = {'data/training.txt': 'training.txt',\n", "            'checkpoints/latest.pt': 'latest.pt',\n", "            'checkpoints/tokenizer.json': 'tokenizer.json',\n", "            'checkpoints/status.json': 'status.json'}\n", 'sources = {}\n', 'for destination, filename in required.items():\n', '    candidates = [DATASET_PATH / filename, DATASET_PATH / destination]\n', '    sources[destination] = next((p for p in candidates if p.is_file()), None)\n', 'missing = [name for name, src in sources.items() if src is None]\n', 'if missing:\n', "    raise FileNotFoundError(f'Missing required dataset files: {missing}. Check DATASET_PATH.')\n", '# Validate all sources first. A rerun must not replace a newer working checkpoint.\n', 'for destination, src in sources.items():\n', '    dst = Path(destination)\n', '    dst.parent.mkdir(parents=True, exist_ok=True)\n', '    if dst.exists():\n', "        print(f'Keeping existing {dst}')\n", '    else:\n', '        shutil.copy2(src, dst)\n', "        print(f'Copied {src.name} -> {dst}')\n", "print('Ready to resume; train.py verifies the checkpoint/corpus hash.')\n"]),

    md('## 6. Train\n\n'
       'The default cell resumes pretraining from the attached checkpoint, preserving its architecture. '
       '`--steps 3000` means 3,000 additional steps; adjust as needed. Other examples:\n\n'
       '```\n'
       '# Continue fine-tuning an existing model:\n'
       '!python train.py --data data/instructions.txt --finetune --steps 3000 --device cuda --wandb\n\n'
       '# Fresh pretraining run from scratch:\n'
       '!python train.py --data data/training.txt --steps 1000 --device cuda --width 768 --layers 8 --heads 12 --wandb\n'
       '```'),
    code(['!python train.py --data data/training.txt --resume --steps 3000 --device cuda --wandb\n']),

    md('## 7. Save your results forward\n\n'
       'Unlike Colab+Drive, this does NOT save automatically. Two options:\n\n'
       '**Quick, one-time:** open the "Output" panel (right side) after this notebook finishes running, '
       'and download `checkpoints/latest.pt`, `tokenizer.json`, `status.json` directly — drop them into '
       'your local `checkpoints/` folder like the Colab workflow.\n\n'
       '**To reuse in a future Kaggle session:** click **Save Version** (top right) with "Save and Run All" '
       'and output saving enabled, then go to your dataset on kaggle.com/datasets and create a **New Version** '
       'from this notebook\'s output — that updates `/kaggle/input/aven-1-data/` for next time, so you don\'t '
       'have to re-upload from your Mac each session.'),
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

out = ROOT / 'Aven-1-Kaggle.ipynb'
out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Wrote {out}')
